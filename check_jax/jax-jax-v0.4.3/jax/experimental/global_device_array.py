# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
import dataclasses
import functools
import numpy as np
from typing import Callable, Sequence, Tuple, Union, Mapping, Optional, List, Dict, NamedTuple

import jax
from jax import core
from jax._src import dispatch
from jax._src import api_util
from jax._src.lib import xla_bridge as xb
from jax._src.lib import xla_client as xc
from jax._src.config import config
from jax._src.interpreters import pxla
from jax.interpreters import xla, mlir
from jax._src.util import prod, safe_zip
from jax._src.interpreters.pxla import PartitionSpec

Shape = Tuple[int, ...]
MeshAxes = PartitionSpec
DeviceArray = xc.Buffer
Device = xc.Device
ArrayLike = Union[np.ndarray, DeviceArray]
Index = Tuple[slice, ...]


_hashed_index = lambda x: hash(tuple((v.start, v.stop) for v in x))


def _get_sharding_spec(global_shape, global_mesh, mesh_axes):
  array_mapping = pxla.get_array_mapping(mesh_axes)
  # The dtype doesn't matter for creating sharding specs.
  aval = core.ShapedArray(global_shape, np.float32)
  return pxla.mesh_sharding_specs(global_mesh.shape,
                                  global_mesh.axis_names)(aval, array_mapping)


def _get_indices(global_shape: Shape, global_mesh: pxla.Mesh,
                 mesh_axes: MeshAxes) -> Tuple[Index, ...]:
  sharding_spec = _get_sharding_spec(global_shape, global_mesh, mesh_axes)
  indices = pxla.spec_to_indices(global_shape, sharding_spec)
  return indices  # type: ignore


@functools.lru_cache(maxsize=4096)
def get_shard_indices(global_shape: Shape, global_mesh: pxla.Mesh,
                      mesh_axes: MeshAxes) -> Mapping[Device, Index]:
  indices = _get_indices(global_shape, global_mesh, mesh_axes)
  # The type: ignore is to ignore the type returned by `spec_to_indices`.
  return {
      d: i
      for d, i in safe_zip(global_mesh.devices.flat, indices)}  # type: ignore


@functools.lru_cache(maxsize=4096)
def get_shard_indices_replica_ids(
    global_shape: Shape, global_mesh: pxla.Mesh,
    mesh_axes: MeshAxes) -> Mapping[Device, Tuple[Index, int]]:
  return _get_shard_indices_replica_ids_uncached(global_shape, global_mesh, mesh_axes)

def _get_shard_indices_replica_ids_uncached(
    global_shape: Shape, global_mesh: pxla.Mesh,
    mesh_axes: MeshAxes) -> Mapping[Device, Tuple[Index, int]]:
  indices = _get_indices(global_shape, global_mesh, mesh_axes)
  index_to_replica: Dict[int, int] = Counter()
  out = {}
  unique_shards = 0
  for device, index in safe_zip(global_mesh.devices.flat, indices):
    h_index = _hashed_index(index)
    replica_id = index_to_replica[h_index]
    if replica_id == 0:
      unique_shards += 1
    index_to_replica[h_index] += 1
    out[device] = (index, replica_id)

  shard_shape = get_shard_shape(global_shape, global_mesh, mesh_axes)
  expected_unique_shards = prod(
      [g // s for g, s in safe_zip(global_shape, shard_shape) if g != 0 or s != 0])
  if expected_unique_shards != unique_shards:
    raise RuntimeError(
        f'Number of expected unique shards are: {expected_unique_shards} but '
        f'got {unique_shards}. Please file a bug at '
        'https://github.com/google/jax/issues.')
  return out


@functools.lru_cache(maxsize=4096)
def get_shard_shape(global_shape, global_mesh, mesh_axes) -> Shape:
  chunk_size = []
  for mesh_axis, size in zip(mesh_axes, global_shape):
    if not mesh_axis:
      chunk_size.append(size)
    elif isinstance(mesh_axis, tuple):
      m = prod([global_mesh.shape[ma] for ma in mesh_axis])
      chunk_size.append(size // m)
    else:
      chunk_size.append(size // global_mesh.shape[mesh_axis])
  if len(chunk_size) != len(global_shape):
    chunk_size.extend(global_shape[len(chunk_size):])
  return tuple(chunk_size)


@dataclasses.dataclass(frozen=True)
class Shard:
  """A single data shard of a GlobalDeviceArray.

  Args:
    device : Which device this shard resides on.
    index : The index into the global array of this shard.
    replica_id : Integer id indicating which replica of the global array this
      shard is part of. Always 0 for fully sharded data
      (i.e. when there’s only 1 replica).
    data : The data of this shard. None if ``device`` is non-local.
  """
  device: Device
  index: Index
  replica_id: int
  # None if this `Shard` lives on a non-local device.
  data: Optional[DeviceArray] = None


class _GdaFastPathArgs(NamedTuple):
  global_indices_replica_ids: Mapping[Device, Tuple[Index, int]]
  local_devices: Sequence[Device]


class GlobalDeviceArray:
  """A logical array with data sharded across multiple devices and processes.

  If you’re not already familiar with JAX’s multi-process programming model,
  please read https://jax.readthedocs.io/en/latest/multi_process.html.
  You can also read about pjit (https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html)
  to learn about ``Mesh``, ``PartitionSpec`` and how arrays can be
  partitioned or replicated.

  A GlobalDeviceArray (GDA) can be thought of as a view into a single logical
  array sharded across processes. The logical array is the “global” array, and
  each process has a GlobalDeviceArray object referring to the same global array
  (similarly to how each process runs a multi-process pmap or pjit). Each process
  can access the shape, dtype, etc. of the global array via the GDA, pass the
  GDA into multi-process pjits, and get GDAs as pjit outputs (coming soon: xmap
  and pmap). However, each process can only directly access the shards of the
  global array data stored on its local devices.

  GDAs can help manage the inputs and outputs of multi-process computations.
  A GDA keeps track of which shard of the global array belongs to which device,
  and provides callback-based APIs to materialize the correct shard of the data
  needed for each local device of each process.

  A GDA consists of data shards. Each shard is stored on a different device.
  There are local shards and global shards. Local shards are those on local
  devices, and the data is visible to the current process. Global shards are
  those across all devices (including local devices), and the data isn’t visible
  if the shard is on a non-local device with respect to the current process.
  Please see the ``Shard`` class to see what information is stored inside that
  data structure.

  Note: to make pjit output GlobalDeviceArrays, set the environment variable
  ``JAX_PARALLEL_FUNCTIONS_OUTPUT_GDA=true`` or add the following to your code:
  ``jax.config.update('jax_parallel_functions_output_gda', True)``

  Args:
    global_shape : The global shape of the array.
    global_mesh : The global mesh representing devices across multiple
      processes.
    mesh_axes : A sequence with length less than or equal to the rank of the
      global array (i.e. the length of the global shape). Each element can be:

      * An axis name of ``global_mesh``, indicating that the corresponding
        global array axis is partitioned across the given device axis of
        ``global_mesh``.
      * A tuple of axis names of ``global_mesh``. This is like the above option
        except the global array axis is partitioned across the product of axes
        named in the tuple.
      * None indicating that the corresponding global array axis is not
        partitioned.

      For more information, please see:
      https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
    device_buffers: DeviceArrays that are on the local devices of ``global_mesh``.

  Attributes:
    shape : Global shape of the array.
    dtype : Dtype of the global array.
    ndim : Number of array dimensions in the global shape.
    size: Number of elements in the global array.
    local_shards : List of :class:`Shard` on the local devices of the current process.
      Data is materialized for all local shards.
    global_shards : List of all :class:`Shard` of the global array. Data isn’t
      available if a shard is on a non-local device with respect to the current
      process.
    is_fully_replicated : True if the full array value is present on all devices
      of the global mesh.

  Example:

    >>> from jax.sharding import Mesh
    >>> from jax.sharding import PartitionSpec as P
    >>> import numpy as np
    ...
    >>> assert jax.device_count() == 8
    >>> global_mesh = Mesh(np.array(jax.devices()).reshape(4, 2), ('x', 'y'))
    >>> # Logical mesh is (hosts, devices)
    >>> assert global_mesh.shape == {'x': 4, 'y': 2}
    >>> global_input_shape = (8, 2)
    >>> mesh_axes = P('x', 'y')
    ...
    >>> # Dummy example data; in practice we wouldn't necessarily materialize global data
    >>> # in a single process.
    >>> global_input_data = np.arange(
    ...   np.prod(global_input_shape)).reshape(global_input_shape)
    ...
    >>> def get_local_data_slice(index):
    ...  # index will be a tuple of slice objects, e.g. (slice(0, 16), slice(0, 4))
    ...  # This method will be called per-local device from the GDA constructor.
    ...  return global_input_data[index]
    ...
    >>> gda = GlobalDeviceArray.from_callback(
    ...        global_input_shape, global_mesh, mesh_axes, get_local_data_slice)
    >>> print(gda.shape)
    (8, 2)
    >>> print(gda.addressable_shards[0].data)  # Access the data on a single local device
    [[0]
     [2]]
    >>> print(gda.addressable_shards[0].data.shape)
    (2, 1)
    >>> # Numpy-style index into the global array that this data shard corresponds to
    >>> print(gda.addressable_shards[0].index)
    (slice(0, 2, None), slice(0, 1, None))

  GDAs can also be given as an input to pjit and you can get GDAs as output from pjit::

    # Allow pjit to output GDAs
    jax.config.update('jax_parallel_functions_output_gda', True)

    f = pjit(lambda x: x @ x.T, in_axis_resources=P('x', 'y'), out_axis_resources = P('x', 'y'))
    with global_mesh:
      out = f(gda)

    # `out` can be passed to another pjit call, out.addressable_shards can be used to
    # export the data to non-jax systems (e.g. for checkpointing or logging), etc.

  """

  def __init__(self,
               global_shape: Shape,
               global_mesh: pxla.Mesh,
               mesh_axes: MeshAxes,
               device_buffers: Union[xb.ShardedBuffer, Sequence[DeviceArray]],
               _gda_fast_path_args: Optional[_GdaFastPathArgs] = None,
               _enable_checks: bool = True):
    self._global_shape = global_shape
    self._global_mesh = global_mesh
    self._mesh_axes = mesh_axes
    self._init_buffers(device_buffers)

    # Optionally precomputed for performance.
    self._gda_fast_path_args = _gda_fast_path_args

    if self._gda_fast_path_args is None:
      self._local_devices = self._global_mesh.local_devices
    else:
      self._local_devices = self._gda_fast_path_args.local_devices

    if _enable_checks or config.jax_enable_checks:
      for db, ld in safe_zip(self._device_buffers, self._local_devices):
        if db.device() != ld:
          raise ValueError(
              "The `global_mesh.local_devices` and `device_buffers` device "
              "order doesn't match. Please use `global_mesh.local_devices` to "
              "put arrays on devices instead of `jax.local_devices()`")

    if _enable_checks or config.jax_enable_checks:
      ss = get_shard_shape(self._global_shape, self._global_mesh, self.mesh_axes)
      assert all(db.shape == ss for db in self._device_buffers), (
          f"Expected shard shape {ss} doesn't match the device buffer "
          f"shape, got: {[db.shape for db in self._device_buffers]}")

    if self._sharded_buffer is None:
      dtype = device_buffers[0].dtype  # type: ignore
    else:
      dtype = self._sharded_buffer.dtype  # type: ignore
    if _enable_checks or config.jax_enable_checks:
      assert all(db.dtype == dtype for db in self._device_buffers), (
          "Input arrays to GlobalDeviceArray must have matching dtypes, "
          f"got: {[db.dtype for db in self._device_buffers]}")
    self.dtype = dtype

  def _init_buffers(self, device_buffers):
    from jax._src.array import ArrayImpl

    self._maybe_device_buffers = None

    # ShardedBuffer is the fast path for managing sharded buffers that avoids
    # creating python objects for every device.
    if isinstance(device_buffers, xc.ShardedBuffer):
      # if ShardedBuffer is provided, we don't need to use `_device_buffers`
      self._sharded_buffer = device_buffers  # type: ignore
    elif isinstance(device_buffers[0], DeviceArray):  # type: ignore
      # if xla_client.Buffer is provided, we convert it to ShardedBuffer.
      self._sharded_buffer = xc.ShardedBuffer.create_sharded_buffer(device_buffers)
    elif isinstance(device_buffers[0], ArrayImpl):
      self._sharded_buffer = None
      self._maybe_device_buffers = [db._arrays[0] for db in device_buffers]
    else:
      # if `device_buffers` is any other types that cannot
      # be converted to ShardedBuffer, then we use `device_buffers`.
      # TODO(yashkatariya,chky): Remove this branch once everyone is using
      # sharded_buffer
      self._sharded_buffer = None
      self._maybe_device_buffers = device_buffers

  @property
  def _device_buffers(self):
    self._check_if_deleted()
    if self._maybe_device_buffers is None:
      self._maybe_device_buffers = self._sharded_buffer.get_device_buffers()  # type: ignore
    return self._maybe_device_buffers

  def __eq__(self, other: object):
    raise NotImplementedError(
        "GlobalDeviceArray equality is intentionally unimplemented. "
        "Implement desired functionality explicitly, e.g. to check if all "
        "values are equal: "
        "pjit(lambda x, y: x == y, "
        "in_axis_resources=FROM_GDA, out_axis_resources=None)"
    )

  def __str__(self):
    return f'GlobalDeviceArray(shape={self.shape}, dtype={self.dtype})'

  def __repr__(self):
    return (f'GlobalDeviceArray(shape={self.shape}, dtype={self.dtype}, '
            f'global_mesh_shape={dict(self.mesh.shape)}, '
            f'mesh_axes={self.mesh_axes})')

  @property
  def shape(self) -> Shape:
    return self._global_shape

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def size(self):
    return prod(self.shape)

  @property
  def mesh(self):
    return self._global_mesh

  @property
  def mesh_axes(self) -> MeshAxes:
    return self._mesh_axes

  @property
  def is_fully_replicated(self) -> bool:
    return self.shape == self.addressable_data(0).shape

  def _create_local_shards(self) -> Sequence[Shard]:
    if self._gda_fast_path_args is not None:
      global_indices_rid = self._gda_fast_path_args.global_indices_replica_ids
    else:
      global_indices_rid = get_shard_indices_replica_ids(
        self._global_shape, self._global_mesh, self.mesh_axes)

    out = []
    for db in self._device_buffers:
      db = dispatch._set_aval(db)
      device = db.device()
      index, rid = global_indices_rid[device]
      out.append(Shard(device, index, rid, db))
    return out

  @functools.cached_property
  def local_shards(self) -> Sequence[Shard]:
    self._check_if_deleted()
    return self._create_local_shards()

  @functools.cached_property
  def addressable_shards(self) -> Sequence[Shard]:
    self._check_if_deleted()
    return self.local_shards

  @property
  def global_shards(self) -> Sequence[Shard]:
    self._check_if_deleted()
    if self.mesh.size == len(self._local_devices):
      return self.addressable_shards

    # Populating global_shards lazily (i.e. when requested) because populating
    # sthem eagerly leads to a performance regression when training on large
    # models.
    # Also as this a cached property, once calculated, it should be cached. So
    # multiple accesses should be cheap.
    global_indices_rid = get_shard_indices_replica_ids(
        self._global_shape, self._global_mesh, self.mesh_axes)
    device_to_buffer = {db.device(): db for db in self._device_buffers}
    global_shards = []
    for device, (index, rid) in global_indices_rid.items():
      local_shard = device.process_index == device.client.process_index()
      buf = device_to_buffer[device] if local_shard else None
      if buf is not None and buf.aval is None:
        buf.aval = core.ShapedArray(buf.shape, buf.dtype)
      sh = Shard(device, index, rid, buf)
      global_shards.append(sh)
    return global_shards

  @property
  def _value(self):
    self._check_if_deleted()
    if self.is_fully_replicated:
      return np.asarray(self._device_buffers[0])

    if self.mesh.is_multi_process:
      raise RuntimeError("Fetching value for GDA that spans non-addressable "
                         "devices is not possible. You can use "
                         "`jax.experimental.multihost_utils.process_allgather` "
                         "for this use case.")
    unique_shards = [s.data.copy_to_host_async() or s
                     for s in self.addressable_shards if s.replica_id == 0]
    npy_value = np.empty(self.shape, self.dtype)
    for s in unique_shards:
      npy_value[s.index] = np.asarray(s.data)
    return npy_value

  def __array__(self, dtype=None, context=None):
    self._check_if_deleted()
    return self._value if dtype is None else self._value.astype(dtype)

  def local_data(self, index) -> DeviceArray:
    self._check_if_deleted()
    return dispatch._set_aval(self._device_buffers[index])

  def addressable_data(self, index) -> DeviceArray:
    self._check_if_deleted()
    return self.local_data(index)

  def block_until_ready(self):
    self._check_if_deleted()
    # self._sharded_buffer can be None if _DeviceArray is used.
    if self._sharded_buffer is None:
      for db in self._device_buffers:
        db.block_until_ready()
    else:
      self._sharded_buffer.block_until_ready()  # type: ignore
    return self

  def _check_if_deleted(self):
    if self.is_deleted():
      raise RuntimeError("GlobalDeviceArray has been deleted.")

  def is_deleted(self):
    return self._sharded_buffer is None and self._maybe_device_buffers is None

  def delete(self):
    if self._sharded_buffer:
      self._sharded_buffer.delete()
      self._sharded_buffer = None
    if self._maybe_device_buffers:
      for b in self._maybe_device_buffers:
        b.delete()
      self._maybe_device_buffers = None

  @property
  def sharding(self):
    return jax.sharding.NamedSharding(self._global_mesh, self.mesh_axes)

  @classmethod
  def from_callback(cls, global_shape: Shape, global_mesh: pxla.Mesh,
                    mesh_axes: MeshAxes, data_callback: Callable[[Index],
                                                                 ArrayLike]):
    """Constructs a GlobalDeviceArray via data fetched from ``data_callback``.

    ``data_callback`` is used to fetch the data for each local slice of the returned GlobalDeviceArray.

    Example:

      >>> from jax.sharding import Mesh
      >>> from jax.sharding import PartitionSpec as P
      >>> import numpy as np
      ...
      >>> global_input_shape = (8, 8)
      >>> mesh_axes = P('x', 'y')
      >>> global_mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ('x', 'y'))
      >>> global_input_data = np.arange(prod(global_input_shape)).reshape(global_input_shape)
      ...
      >>> def cb(index):
      ...  return global_input_data[index]
      ...
      >>> gda = GlobalDeviceArray.from_callback(global_input_shape, global_mesh, mesh_axes, cb)
      >>> gda.addressable_data(0).shape
      (4, 2)

    Args:
      global_shape : The global shape of the array
      global_mesh : The global mesh representing devices across multiple
        processes.
      mesh_axes : See the ``mesh_axes`` parameter of GlobalDeviceArray.
      data_callback : Callback that takes indices into the global array value as input and
        returns the corresponding data of the global array value.  The data can be returned
        as any array-like object, e.g. a ``numpy.ndarray``.
    """
    global_indices_rid = get_shard_indices_replica_ids(
        global_shape, global_mesh, mesh_axes)
    local_devices = global_mesh.local_devices
    dbs = [
        jax.device_put(data_callback(global_indices_rid[device][0]), device)
        for device in local_devices
    ]
    return cls(global_shape, global_mesh, mesh_axes, dbs,
               _gda_fast_path_args=_GdaFastPathArgs(global_indices_rid, local_devices))

  @classmethod
  def from_batched_callback(cls, global_shape: Shape,
                            global_mesh: pxla.Mesh, mesh_axes: MeshAxes,
                            data_callback: Callable[[Sequence[Index]],
                                                    Sequence[ArrayLike]]):
    """Constructs a GlobalDeviceArray via batched data fetched from ``data_callback``.

    Like ``from_callback``, except the callback function is called only once to fetch all data
    local to this process.

    Example:

      >>> from jax.experimental.maps import Mesh
      >>> from jax.experimental import PartitionSpec as P
      >>> import numpy as np
      ...
      >>> global_input_shape = (8, 2)
      >>> mesh_axes = P('x')
      >>> global_mesh = Mesh(np.array(jax.devices()).reshape(4, 2), ('x', 'y'))
      >>> global_input_data = np.arange(prod(global_input_shape)).reshape(global_input_shape)
      ...
      >>> def batched_cb(indices):
      ...   assert len(indices) == len(global_mesh.local_devices)
      ...   return [global_input_data[index] for index in indices]
      ...
      >>> gda = GlobalDeviceArray.from_batched_callback(global_input_shape, global_mesh, mesh_axes, batched_cb)
      >>> gda.addressable_data(0).shape
      (2, 2)

    Args:
      global_shape : The global shape of the array
      global_mesh : The global mesh representing devices across multiple
        processes.
      mesh_axes : See the ``mesh_axes`` parameter of GlobalDeviceArray.
      data_callback : Callback that takes a batch of indices into the global array value with
        length equal to the number of local devices as input and returns the corresponding data for each index.
        The data can be returned as any array-like objects, e.g. ``numpy.ndarray``
    """
    global_indices_rid = get_shard_indices_replica_ids(
        global_shape, global_mesh, mesh_axes)
    local_devices = global_mesh.local_devices
    local_indices = [global_indices_rid[d][0] for d in local_devices]
    local_arrays = data_callback(local_indices)
    dbs = pxla.device_put(local_arrays, local_devices)
    return cls(global_shape, global_mesh, mesh_axes, dbs,
               _gda_fast_path_args=_GdaFastPathArgs(global_indices_rid, local_devices))

  @classmethod
  def from_batched_callback_with_devices(
      cls, global_shape: Shape, global_mesh: pxla.Mesh,
      mesh_axes: MeshAxes,
      data_callback: Callable[[Sequence[Tuple[Index, Tuple[Device, ...]]]],
                              Sequence[DeviceArray]]):
    """Constructs a GlobalDeviceArray via batched DeviceArrays fetched from ``data_callback``.

    Like ``from_batched_callback``, except the callback function is responsible for returning on-device data (e.g. by calling ``jax.device_put``).

    Example:

      >>> from jax.experimental.maps import Mesh
      >>> from jax.experimental import PartitionSpec as P
      >>> import numpy as np
      ...
      >>> global_input_shape = (8, 2)
      >>> mesh_axes = P(('x', 'y'))
      >>> global_mesh = Mesh(np.array(jax.devices()).reshape(4, 2), ('x', 'y'))
      >>> global_input_data = np.arange(prod(global_input_shape)).reshape(global_input_shape)
      ...
      >>> def cb(cb_inp):
      ...  dbs = []
      ...  for inp in cb_inp:
      ...    index, devices = inp
      ...    array = global_input_data[index]
      ...    dbs.extend([jax.device_put(array, device) for device in devices])
      ...  return dbs
      ...
      >>> gda = GlobalDeviceArray.from_batched_callback_with_devices(
      ...   global_input_shape, global_mesh, mesh_axes, cb)
      >>> gda.addressable_data(0).shape
      (1, 2)

    Args:
      global_shape : The global shape of the array
      global_mesh : The global mesh representing devices across multiple
        processes.
      mesh_axes : See the ``mesh_axes`` parameter of GlobalDeviceArray.
      data_callback : Callback that takes agets batch of indices into the global array value with
        length equal to the number of local devices as input and returns the corresponding data for
        each index. The data must be returned as jax DeviceArrays.
    """
    global_indices_rid = get_shard_indices_replica_ids(
        global_shape, global_mesh, mesh_axes)
    local_devices = global_mesh.local_devices

    index_to_device: Dict[int, Tuple[Index, List[Device]]] = {}
    for device in local_devices:
      index = global_indices_rid[device][0]
      h_index = _hashed_index(index)
      if h_index not in index_to_device:
        index_to_device[h_index] = (index, [device])
      else:
        index_to_device[h_index][1].append(device)

    cb_inp = [
        (index, tuple(devices)) for index, devices in index_to_device.values()
    ]
    dbs = data_callback(cb_inp)
    return cls(global_shape, global_mesh, mesh_axes, dbs,
               _gda_fast_path_args=_GdaFastPathArgs(global_indices_rid, local_devices))


core.pytype_aval_mappings[GlobalDeviceArray] = lambda x: core.ShapedArray(
    x.shape, x.dtype)
xla.pytype_aval_mappings[GlobalDeviceArray] = lambda x: core.ShapedArray(
    x.shape, x.dtype)
xla.canonicalize_dtype_handlers[GlobalDeviceArray] = pxla.identity
api_util._shaped_abstractify_handlers[GlobalDeviceArray] = \
    lambda x: core.ShapedArray(x.shape, x.dtype)

# This will only work when GDA is fully addressable i.e. on a single host or
# fully replicated.
def _gda_mlir_constant_handler(val, canonicalize_types=True):
  return mlir.ir_constants(val._value,
                           canonicalize_types=canonicalize_types)
mlir.register_constant_handler(GlobalDeviceArray, _gda_mlir_constant_handler)


def _gda_shard_arg(x, devices, indices):
  x._check_if_deleted()
  # self._sharded_buffer can be None if _DeviceArray is used.
  if x._sharded_buffer is None:
    return x._device_buffers
  return x._sharded_buffer
pxla.shard_arg_handlers[GlobalDeviceArray] = _gda_shard_arg


def _gda_array_result_handler(global_aval, out_sharding, committed,
                              is_out_sharding_from_xla):
  if core.is_opaque_dtype(global_aval.dtype):
    return global_aval.dtype._rules.global_sharded_result_handler(
        global_aval, out_sharding, committed, is_out_sharding_from_xla)
  global_mesh, out_axis_resources = out_sharding.mesh, out_sharding.spec
  global_idx_rid = get_shard_indices_replica_ids(global_aval.shape, global_mesh,
                                                 out_axis_resources)
  local_devices = global_mesh.local_devices
  fast_path_args = _GdaFastPathArgs(global_idx_rid, local_devices)
  return lambda bufs: GlobalDeviceArray(
      global_aval.shape, global_mesh, out_axis_resources, bufs, fast_path_args,
      _enable_checks=False)
pxla.global_result_handlers[
    (core.ShapedArray, pxla.OutputType.GlobalDeviceArray)] = _gda_array_result_handler
pxla.global_result_handlers[
    (core.ConcreteArray, pxla.OutputType.GlobalDeviceArray)] = _gda_array_result_handler
