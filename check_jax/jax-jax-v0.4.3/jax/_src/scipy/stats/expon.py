# Copyright 2018 The JAX Authors.
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

import scipy.stats as osp_stats

from jax import lax
from jax._src.numpy.util import _wraps
from jax._src.numpy.lax_numpy import _promote_args_inexact, where, inf
from jax._src.typing import Array, ArrayLike


@_wraps(osp_stats.expon.logpdf, update_doc=False)
def logpdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, loc, scale = _promote_args_inexact("expon.logpdf", x, loc, scale)
  log_scale = lax.log(scale)
  linear_term = lax.div(lax.sub(x, loc), scale)
  log_probs = lax.neg(lax.add(linear_term, log_scale))
  return where(lax.lt(x, loc), -inf, log_probs)

@_wraps(osp_stats.expon.pdf, update_doc=False)
def pdf(x: ArrayLike, loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, loc, scale))
