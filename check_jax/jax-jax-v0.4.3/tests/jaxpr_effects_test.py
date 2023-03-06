# Copyright 2022 The JAX Authors.
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
import functools
import threading
import unittest
import warnings

from absl.testing import absltest
import jax
import jax.numpy as jnp
from jax import core
from jax import lax
from jax._src import linear_util as lu
from jax.config import config
from jax.interpreters import ad
from jax.experimental import maps
from jax.experimental import pjit
from jax._src import sharding
from jax.interpreters import mlir
from jax._src import ad_checkpoint
from jax._src import dispatch
from jax._src import test_util as jtu
from jax._src import util
from jax._src.lax import control_flow as lcf
from jax._src.lib import xla_bridge
import numpy as np

config.parse_flags_with_absl()

effect_p = core.Primitive('effect')
effect_p.multiple_results = True

@effect_p.def_effectful_abstract_eval
def _(*avals, effect):
  return avals, {effect}

def effect_jvp_rule(primals, tangents, effect):
  return effect_p.bind(*primals, effect=effect), tangents
ad.primitive_jvps[effect_p] = effect_jvp_rule

mlir.lowerable_effects.add('foo')
mlir.lowerable_effects.add('foo2')
mlir.lowerable_effects.add('bar')
mlir.lowerable_effects.add('while')
mlir.lowerable_effects.add('while1')
mlir.lowerable_effects.add('while2')
core.ordered_effects.add('foo')
core.ordered_effects.add('foo2')
core.ordered_effects.add('while1')
core.ordered_effects.add('while2')

lcf.allowed_effects.add('while')
lcf.allowed_effects.add('while1')
lcf.allowed_effects.add('while2')

ad_checkpoint.remat_allowed_effects.add('remat')


def trivial_effect_lowering(ctx, *, effect):
  ctx.set_tokens_out(ctx.tokens_in)
  return []
mlir.register_lowering(effect_p, trivial_effect_lowering)

def function_effect_lowering(ctx, *, effect):
  def _f(ctx):
    ctx.set_tokens_out(ctx.tokens_in)
    return []
  func = mlir._emit_lowering_rule_as_fun(_f, ctx)

  output_types = map(mlir.aval_to_ir_types, ctx.avals_out)
  token_types = [mlir.token_type() for _ in ctx.tokens_in.items()]
  output_types = [*token_types, *output_types]
  flat_output_types = util.flatten(output_types)
  call = mlir.func_dialect.CallOp(flat_output_types,
                                  mlir.ir.FlatSymbolRefAttr.get(func.name.value),
                                  mlir.flatten_lowering_ir_args(ctx.tokens_in.tokens()))
  tokens, out = util.split_list(call.results, [len(ctx.tokens_in)])
  ctx.set_tokens_out(mlir.TokenSet(zip(ctx.tokens_in.effects(), tokens)))
  return out

callback_p = core.Primitive('callback')
callback_p.multiple_results = True

mlir.lowerable_effects.add('log')
mlir.lowerable_effects.add('unordered_log')
core.ordered_effects.add('log')

@callback_p.def_impl
def _(*args, callback, out_avals, effect):
  del out_avals, effect
  callback(*args)
  return []

@callback_p.def_effectful_abstract_eval
def _(*avals, callback, out_avals, effect):
  del avals, callback
  return out_avals, {effect}

def callback_effect_lowering(ctx: mlir.LoweringRuleContext, *args, callback, out_avals, effect):
  del out_avals
  token_in = None
  if effect in core.ordered_effects:
    token_in = ctx.tokens_in.get(effect)[0]

  out_op, token_out, keep_alive = mlir.emit_python_callback(
      ctx, callback, token_in, list(args), list(ctx.avals_in),
      list(ctx.avals_out), True)
  if token_out:
    ctx.set_tokens_out(ctx.tokens_in.update_tokens(mlir.TokenSet({effect:
      token_out})))
  ctx.module_context.add_keepalive(keep_alive)
  return out_op

mlir.register_lowering(callback_p, callback_effect_lowering)


prev_xla_flags = None


def setUpModule():
  global prev_xla_flags
  # This will control the CPU devices. On TPU we always have 2 devices
  prev_xla_flags = jtu.set_host_platform_device_count(2)


# Reset to previous configuration in case other test modules will be run.
def tearDownModule():
  prev_xla_flags()


class JaxprEffectsTest(jtu.JaxTestCase):

  def test_trivial_jaxpr_has_no_effects(self):
    def f(x):
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual(core.no_effects, jaxpr.effects)

  def test_effectful_primitive_in_jaxpr_creates_effects(self):
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual({'foo'}, jaxpr.jaxpr.eqns[0].effects)
    self.assertEqual({'foo'}, jaxpr.effects)

  def test_different_effects_in_jaxpr(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertEqual({'foo'}, jaxpr.jaxpr.eqns[0].effects)
    self.assertEqual({'bar'}, jaxpr.jaxpr.eqns[1].effects)
    self.assertEqual({'foo', 'bar'}, jaxpr.effects)

  def test_jaxpr_typecheck_should_verify_eqn_effects_are_subset(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x + 1.
    jaxpr = jax.make_jaxpr(f)(2.).jaxpr

    # Edit jaxpr to make its type wrong
    jaxpr = jaxpr.replace(effects={'foo'})

    with self.assertRaisesRegex(core.JaxprTypeError,
        'Equation effects are not subset of Jaxpr effects.'):
      core.check_jaxpr(jaxpr)

class HigherOrderPrimitiveTest(jtu.JaxTestCase):

  def test_core_call_primitive_inherits_effects(self):

    def f(x):
      @lu.wrap_init
      def f_(x):
        effect_p.bind(effect='foo')
        effect_p.bind(effect='bar')
        return [x]
      return core.call(f_, x)[0]
    jax.make_jaxpr(f)(2.)

  def test_xla_call_primitive_inherits_effects(self):

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    jax.make_jaxpr(f)(2.)

  @jtu.sample_product(flavor=["old", "new"])
  def test_remat_call_primitive_inherits_effects(self, flavor):
    remat = jax.remat if flavor == "old" else ad_checkpoint.checkpoint

    @remat
    def f(x):
      x, = effect_p.bind(x, effect='foo')
      x, = effect_p.bind(x, effect='bar')
      return x
    jax.make_jaxpr(f)(2.)
    with self.assertRaisesRegex(NotImplementedError, "Effects not supported"):
      jax.make_jaxpr(lambda x: jax.linearize(f, x)[1](x))(2.)

  def test_new_remat_allows_certain_effects(self):
    @ad_checkpoint.checkpoint
    def f(x):
      x, = effect_p.bind(x, effect='remat')
      return x
    jaxpr = jax.make_jaxpr(f)(2.)
    self.assertSetEqual(jaxpr.effects, {"remat"})

  def test_custom_jvp_primitive_inherits_effects(self):

    @jax.custom_jvp
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    f.defjvp(lambda x, t: (x, t))
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

  def test_custom_vjp_primitive_inherits_effects(self):

    @jax.custom_vjp
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    f.defvjp(
        fwd=lambda x: (x, ()),
        bwd=lambda _, g: g)
    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

  def test_pmap_inherits_effects(self):

    @jax.pmap
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    with self.assertRaisesRegex(
        ValueError,
        "Ordered effects not supported for map primitives: {'foo'}"):
      jax.make_jaxpr(f)(jnp.arange(jax.local_device_count()))

  def test_xmap_inherits_effects(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    f = maps.xmap(f, in_axes=['a'], out_axes=['a'])
    jaxpr = jax.make_jaxpr(f)(jnp.arange(jax.local_device_count()))
    self.assertSetEqual(jaxpr.effects, {"foo", "bar"})

  def test_pjit_inherits_effects(self):
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='bar')
      return x
    mesh = jax.sharding.Mesh(np.array(jax.devices()), ['x'])
    if config.jax_array:
      spec = sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x'))
    else:
      spec = jax.sharding.PartitionSpec('x')
    f = pjit.pjit(f, in_axis_resources=spec, out_axis_resources=spec)
    with mesh:
      jaxpr = jax.make_jaxpr(f)(np.arange(jax.local_device_count()))
    self.assertSetEqual(jaxpr.effects, {"foo", "bar"})


class EffectfulJaxprLoweringTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    self.old_x64 = config.jax_enable_x64
    config.update('jax_enable_x64', False)
    self._old_lowering = mlir._lowerings[effect_p]
    def _effect_lowering(ctx, *, effect):
      if effect in core.ordered_effects:
        expected_effects = [effect]
      else:
        expected_effects = []
      self.assertListEqual(list(ctx.tokens_in.effects()), expected_effects)
      ctx.set_tokens_out(ctx.tokens_in)
      return []
    mlir.register_lowering(effect_p, _effect_lowering)
    jax.effects_barrier()
    dispatch.runtime_tokens.clear()

  def tearDown(self):
    super().tearDown()
    dispatch.runtime_tokens.clear()
    config.update('jax_enable_x64', self.old_x64)
    mlir.register_lowering(effect_p, self._old_lowering)

  def test_cannot_lower_unlowerable_effect(self):
    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    f.lower(2.)

  def test_should_not_pass_tokens_into_unordered_effect(self):

    def effect_lowering(ctx, *, effect):
      self.assertEmpty(ctx.tokens_in)
      return []
    mlir.register_lowering(effect_p, effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='bar')
      return x + 1.
    f.lower(2.)

  def test_lowering_that_doesnt_set_tokens_should_cause_error(self):

    def bad_effect_lowering(ctx, *, effect):
      # Doesn't call `ctx.set_tokens_out`!
      return []
    mlir.register_lowering(effect_p, bad_effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    with self.assertRaisesRegex(ValueError, 'Lowering rule for `effect` needs to '
        'set `tokens_out`'):
      f.lower(2.)

  def test_lowering_that_sets_wrong_tokens_should_cause_error(self):

    def bad_effect_lowering(ctx, *, effect):
      ctx.set_tokens_out(mlir.TokenSet(bar=ctx.tokens_in.get('foo')))
      return []
    mlir.register_lowering(effect_p, bad_effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    with self.assertRaisesRegex(ValueError, 'Lowering rule for `effect` returns '
        'incorrect set of output token.'):
      f.lower(2.)

  def test_lowering_ordered_effect_should_create_tokens(self):

    def effect_lowering(ctx, *, effect):
      ctx.set_tokens_out(ctx.tokens_in)
      return []
    mlir.register_lowering(effect_p, effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    module = f.lower(2.).compiler_ir()
    main = module.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertIn('hlo.create_token', first_op.operation.name)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='foo2')
      return x + 1.
    module = f.lower(2.).compiler_ir()
    main = module.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertIn('hlo.create_token', first_op.operation.name)
    second_op = main.body.blocks[0].operations[1]
    self.assertIn('hlo.create_token', second_op.operation.name)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    module = f.lower(2.).compiler_ir()
    main = module.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertIn('hlo.create_token', first_op.operation.name)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='foo2')
      return x + 1.
    module = f.lower(2.).compiler_ir()
    main = module.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertIn('hlo.create_token', first_op.operation.name)
    second_op = main.body.blocks[0].operations[1]
    self.assertIn('hlo.create_token', second_op.operation.name)

  def test_nontrivial_lowering_with_ordered_effect_should_consume_token(self):

    mlir.register_lowering(effect_p, function_effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    module = f.lower(2.).compiler_ir()
    main = module.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertIn('hlo.create_token', first_op.operation.name)
    second_op = main.body.blocks[0].operations[1]
    self.assertEqual(second_op.operation.name, "func.call")
    self.assertEqual(str(second_op.attributes["callee"]), "@effect")
    self.assertEqual(second_op.operands[0].owner, first_op)
    func = module.body.operations[1]
    self.assertEqual(func.name.value, "effect")
    self.assertIn('hlo.token', str(func.type.inputs[0]))
    self.assertIn('hlo.token', str(func.type.results[0]))

  def test_nontrivial_lowering_with_unordered_effect_should_consume_token(self):

    mlir.register_lowering(effect_p, function_effect_lowering)

    @jax.jit
    def f(x):
      effect_p.bind(effect='bar')
      return x + 1.
    module = f.lower(2.).compiler_ir()
    main = module.body.operations[0]
    first_op = main.body.blocks[0].operations[0]
    self.assertEqual(first_op.operation.name, "func.call")
    self.assertEqual(str(first_op.attributes["callee"]), "@effect")
    self.assertLen(list(first_op.operands), 0)
    func = module.body.operations[1]
    self.assertEqual(func.name.value, "effect")
    self.assertLen(list(func.type.inputs), 0)
    self.assertLen(list(func.type.results), 0)

  def test_lowered_jaxpr_without_ordered_effects_takes_no_dummy_inputs(self):
    @jax.jit
    def f(x):
      effect_p.bind(effect='bar')
      return x + 1.
    module = f.lower(1.).compiler_ir()
    input_types = module.body.operations[0].type.inputs
    self.assertLen(list(input_types), 1)
    self.assertEqual(str(input_types[0]), 'tensor<f32>')

    # First output should be output token
    result_types = module.body.operations[0].type.results
    self.assertLen(list(result_types), 1)
    self.assertEqual(str(result_types[0]), 'tensor<f32>')

  def test_lowered_jaxpr_with_ordered_effects_takes_in_dummy_inputs(self):
    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    module = f.lower(1.).compiler_ir()
    input_types = module.body.operations[0].type.inputs
    # First argument should be dummy token
    self.assertLen(list(input_types), 2)
    self.assertEqual(str(input_types[0]), 'tensor<0xi1>')

    # First output should be dummy token
    result_types = module.body.operations[0].type.results
    self.assertLen(list(result_types), 2)
    self.assertEqual(str(result_types[0]), 'tensor<0xi1>')

  def test_lowered_jaxpr_with_multiple_ordered_effects_takes_in_dummy_inputs(self):
    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='foo2')
      return x + 1.
    module = f.lower(1.).compiler_ir()
    input_types = module.body.operations[0].type.inputs
    # First two arguments should be dummy values
    self.assertLen(list(input_types), 3)
    self.assertEqual(str(input_types[0]), 'tensor<0xi1>')
    self.assertEqual(str(input_types[1]), 'tensor<0xi1>')

    # First two outputs should be dummy values
    result_types = module.body.operations[0].type.results
    self.assertLen(list(result_types), 3)
    self.assertEqual(str(result_types[0]), 'tensor<0xi1>')
    self.assertEqual(str(result_types[1]), 'tensor<0xi1>')

  def test_can_lower_and_run_jaxpr_with_ordered_effects(self):
    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    self.assertEqual(f(2.), 3.)

  def test_can_lower_and_run_jaxpr_with_unordered_effects(self):
    @jax.jit
    def f(x):
      effect_p.bind(effect='bar')
      return x + 1.
    self.assertEqual(f(2.), 3.)

  def test_cant_jit_and_pmap_function_with_unordered_effects(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")
    @jax.jit
    @jax.pmap
    def f(x):
      effect_p.bind(effect='bar')
      return x + 1
    with self.assertRaisesRegex(
        NotImplementedError,
        "Cannot execute replicated computation with effects."):
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f(jnp.arange(jax.device_count()))

  def test_cant_jit_and_pmap_function_with_ordered_effects(self):
    @jax.jit
    @jax.pmap
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1
    with self.assertRaisesRegex(
        ValueError,
        "Ordered effects not supported for map primitives: {'foo'}"):
      f(jnp.arange(jax.device_count()))

  def test_runtime_tokens_should_update_after_running_effectful_function(self):
    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      return x + 1.
    self.assertNotIn('foo', dispatch.runtime_tokens.tokens)
    f(2.)
    prev_token = dispatch.runtime_tokens.tokens['foo']
    f(2.)
    curr_token = dispatch.runtime_tokens.tokens['foo']
    self.assertIsNot(prev_token, curr_token)

  def test_can_lower_multiple_effects(self):
    @jax.jit
    def f(x):
      effect_p.bind(effect='foo')
      effect_p.bind(effect='foo2')
      return x + 1.
    @jax.jit
    def g(x):
      effect_p.bind(effect='foo')
      return x + 1.
    self.assertNotIn('foo', dispatch.runtime_tokens.tokens)
    self.assertNotIn('foo2', dispatch.runtime_tokens.tokens)
    f(2.)
    foo_token = dispatch.runtime_tokens.tokens['foo'][0]
    foo2_token = dispatch.runtime_tokens.tokens['foo'][0]
    f(2.)
    self.assertIsNot(foo_token, dispatch.runtime_tokens.tokens['foo'][0])
    self.assertIsNot(foo2_token, dispatch.runtime_tokens.tokens['foo2'][0])
    foo_token = dispatch.runtime_tokens.tokens['foo'][0]
    foo2_token = dispatch.runtime_tokens.tokens['foo2'][0]
    g(2.)
    self.assertIsNot(foo_token, dispatch.runtime_tokens.tokens['foo'][0])
    self.assertIs(foo2_token, dispatch.runtime_tokens.tokens['foo2'][0])

@jtu.pytest_mark_if_available('pjrt_c_api_unimplemented')  # host callback
class EffectOrderingTest(jtu.JaxTestCase):

  def test_can_execute_python_callback(self):
    # TODO(sharadmv): enable this test on GPU and TPU when backends are
    # supported
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    log = []
    def log_value(x):
      log.append(x)
      return ()

    @jax.jit
    def f(x):
      return callback_p.bind(x, callback=log_value, effect='log', out_avals=[])

    f(2.)
    jax.effects_barrier()
    self.assertListEqual(log, [2.])
    f(3.)
    jax.effects_barrier()
    self.assertListEqual(log, [2., 3.])

  @jtu.skip_on_devices("tpu")
  def test_ordered_effect_remains_ordered_across_multiple_devices(self):
    # TODO(sharadmv): enable this test on GPU and TPU when backends are
    # supported
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")

    log = []
    def log_value(x):
      log.append(x)
      return ()

    @functools.partial(jax.jit, device=jax.devices()[0])
    def f(x):
      # Expensive computation
      x = x.dot(x)
      x = jnp.log(x.sum())
      return callback_p.bind(x, callback=log_value, effect='log', out_avals=[])

    @functools.partial(jax.jit, device=jax.devices()[1])
    def g(x):
      return callback_p.bind(x, callback=log_value, effect='log', out_avals=[])

    f(jnp.ones((500, 500)))
    g(3.)
    f(jnp.ones((500, 500)))
    g(3.)
    f(jnp.ones((500, 500)))
    g(3.)
    jax.effects_barrier()
    x_, y_ = float(jnp.log(1.25e8)), 3.
    expected_log = [x_, y_, x_, y_, x_, y_]
    self.assertListEqual(log, expected_log)

  @jtu.skip_on_devices("tpu")
  def test_different_threads_get_different_tokens(self):
    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")
    tokens = []
    def _noop(_):
      tokens.append(dispatch.runtime_tokens.tokens['log'][0])
      return ()

    @functools.partial(jax.jit, device=jax.devices()[0])
    def f(x):
      return callback_p.bind(x, callback=_noop, effect='log', out_avals=[])

    t1 = threading.Thread(target=lambda: f(2.))
    t2 = threading.Thread(target=lambda: f(3.))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    token1, token2 = tokens
    self.assertIsNot(token1, token2)

class ParallelEffectsTest(jtu.JaxTestCase):

  def test_cannot_pmap_unlowerable_effect(self):

    def f(x):
      # abc is not lowerable
      effect_p.bind(effect='abc')
      return x
    with self.assertRaisesRegex(
        ValueError, "Cannot lower jaxpr with effects: {'abc'}"):
      jax.pmap(f)(jnp.arange(jax.local_device_count()))

  def test_cannot_pmap_ordered_effect(self):

    def f(x):
      # foo is lowerable and ordered
      effect_p.bind(effect='foo')
      return x
    with self.assertRaisesRegex(
        ValueError, "Ordered effects not supported in `pmap`."):
      jax.pmap(f)(jnp.arange(jax.local_device_count()))

  def test_can_pmap_unordered_effect(self):

    def f(x):
      # bar is lowerable and unordered
      effect_p.bind(effect='bar')
      return x
    jax.pmap(f)(jnp.arange(jax.local_device_count()))

  @jtu.pytest_mark_if_available('pjrt_c_api_unimplemented')  # host callback
  def test_can_pmap_unordered_callback(self):
    # TODO(sharadmv): enable this test on GPU and TPU when backends are
    # supported
    if xla_bridge.get_backend().runtime_type == 'stream_executor':
      raise unittest.SkipTest('Host callback not supported for runtime type: stream_executor.')

    if jax.device_count() < 2:
      raise unittest.SkipTest("Test requires >= 2 devices.")

    log = set()
    def log_value(x):
      log.add(int(x))
      return ()

    @jax.pmap
    def f(x):
      callback_p.bind(
          x, callback=log_value, effect='unordered_log', out_avals=[])
      return x + 1
    f(jnp.arange(2)).block_until_ready()
    jax.effects_barrier()
    self.assertSetEqual({0, 1}, log)

class ControlFlowEffectsTest(jtu.JaxTestCase):

  def test_effects_disallowed_in_cond(self):
    def f1(x):
      def true_fun(x):
        effect_p.bind(effect='foo')
        return x
      def false_fun(x):
        return x
      return lax.cond(True, true_fun, false_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f1)(2.)

  def test_allowed_effect_in_cond(self):
    def f(x):
      def true_fun(x):
        effect_p.bind(effect='while')
        return x
      def false_fun(x):
        effect_p.bind(effect='while')
        return x
      return lax.cond(x, true_fun, false_fun, x)
    f(2)

  def test_allowed_effect_in_cond_jvp(self):
    def f(x):
      def true_fun(x):
        effect_p.bind(effect='while')
        return x
      def false_fun(x):
        effect_p.bind(effect='while')
        return x
      return lax.cond(True, true_fun, false_fun, x)

    # test primal side gets effect
    primal_jaxpr = jax.make_jaxpr(lambda x: jax.linearize(f, x)[0])(2.)
    self.assertEqual(primal_jaxpr.effects, {'while'})
    # and tangent side does not
    _, f_lin = jax.linearize(f, 2.)
    lin_jaxpr = f_lin.func.fun.args[0]
    self.assertEqual(lin_jaxpr.effects, set())

  def test_allowed_effect_in_cond_jvp2(self):
    @jax.custom_jvp
    def print_tangents(x):
      return x
    @print_tangents.defjvp
    def foo_jvp(primals, tangents):
      x, = primals
      t, = tangents
      # TODO(mattjj,sharadmv): don't require data dependence for jax.linearize!
      # effect_p.bind(t, effect='while')
      t, = effect_p.bind(t, effect='while')  # data dep only on tangents
      return x, t

    def f(x):
      def true_fun(x):
        return print_tangents(x)
      def false_fun(x):
        return print_tangents(x)
      return lax.cond(True, true_fun, false_fun, x)

    # test primal side does not get effect
    primal_jaxpr = jax.make_jaxpr(lambda x: jax.linearize(f, x)[0])(2.)
    self.assertEqual(primal_jaxpr.effects, set())
    # and tangent side does
    _, f_lin = jax.linearize(f, 2.)
    lin_jaxpr = f_lin.func.fun.args[0]
    self.assertEqual(lin_jaxpr.effects, {'while'})

  def test_allowed_ordered_effect_in_cond(self):
    def f(x):
      def true_fun(x):
        effect_p.bind(effect='while1')
        return x
      def false_fun(x):
        effect_p.bind(effect='while1')
        return x
      return lax.cond(x, true_fun, false_fun, x)
    f(2)

  def test_multiple_allowed_ordered_effect_in_cond(self):
    def f(x):
      def true_fun(x):
        effect_p.bind(effect='while1')
        effect_p.bind(effect='while2')
        return x
      def false_fun(x):
        effect_p.bind(effect='while1')
        effect_p.bind(effect='while2')
        return x
      return lax.cond(x, true_fun, false_fun, x)
    f(2)

    def f2(x):
      def true_fun(x):
        return x
      def false_fun(x):
        effect_p.bind(effect='foo')
        return x
      return lax.cond(True, true_fun, false_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f2)(2.)

  def test_allowed_effect_in_while_body(self):
    def f(x):
      def cond_fun(x):
        return False
      def body_fun(x):
        effect_p.bind(effect='while')
        return x
      return lax.while_loop(cond_fun, body_fun, x)
    f(2)

  def test_allowed_effect_in_cond_body(self):
    def f(x):
      def cond_fun(x):
        effect_p.bind(effect='while')
        return False
      def body_fun(x):
        return x
      return lax.while_loop(cond_fun, body_fun, x)
    f(2)

  def test_allowed_ordered_effect_in_while_body(self):
    def f(x):
      def cond_fun(x):
        return False
      def body_fun(x):
        effect_p.bind(effect='while1')
        return x
      return lax.while_loop(cond_fun, body_fun, x)
    f(2)

  def test_multiple_allowed_ordered_effect_in_while_body(self):
    def f(x):
      def cond_fun(x):
        return False
      def body_fun(x):
        effect_p.bind(effect='while1')
        effect_p.bind(effect='while2')
        return x
      return lax.while_loop(cond_fun, body_fun, x)
    f(2)

  def test_effects_disallowed_in_while(self):
    def f1(x):
      def cond_fun(x):
        effect_p.bind(effect='foo')
        return False
      def body_fun(x):
        return x
      return lax.while_loop(cond_fun, body_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f1)(2.)

    def f2(x):
      def cond_fun(x):
        return False
      def body_fun(x):
        effect_p.bind(effect='foo')
        return x
      return lax.while_loop(cond_fun, body_fun, x)

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f2)(2.)

  def test_allowed_effect_in_scan(self):
    def f(x):
      def body_fun(carry, x):
        effect_p.bind(effect='while')
        return carry, x
      return lax.scan(body_fun, x, jnp.arange(5))
    f(2)

  def test_allowed_ordered_effect_in_scan(self):
    def f(x):
      def body_fun(carry, x):
        effect_p.bind(effect='while1')
        return carry, x
      return lax.scan(body_fun, x, jnp.arange(5))
    f(2)

  def test_multiple_allowed_ordered_effect_in_scan(self):
    def f(x):
      def body_fun(carry, x):
        effect_p.bind(effect='while1')
        effect_p.bind(effect='while2')
        return carry, x
      return lax.scan(body_fun, x, jnp.arange(5))
    f(2)

  def test_effects_disallowed_in_scan(self):

    def f(x):
      def body(carry, x):
        effect_p.bind(effect='foo')
        return carry, x
      return lax.scan(body, x, jnp.arange(4))

    with self.assertRaisesRegex(NotImplementedError, 'Effects not supported'):
      jax.make_jaxpr(f)(2.)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
