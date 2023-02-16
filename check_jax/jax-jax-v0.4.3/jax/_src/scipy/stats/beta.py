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
from jax._src.lax.lax import _const as _lax_const
from jax._src.numpy.util import _wraps
from jax._src.numpy.lax_numpy import _promote_args_inexact, where, inf, logical_or
from jax._src.typing import Array, ArrayLike
from jax.scipy.special import betaln, xlogy, xlog1py


@_wraps(osp_stats.beta.logpdf, update_doc=False)
def logpdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
           loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  x, a, b, loc, scale = _promote_args_inexact("beta.logpdf", x, a, b, loc, scale)
  one = _lax_const(x, 1)
  shape_term = lax.neg(betaln(a, b))
  y = lax.div(lax.sub(x, loc), scale)
  log_linear_term = lax.add(xlogy(lax.sub(a, one), y),
                            xlog1py(lax.sub(b, one), lax.neg(y)))
  log_probs = lax.sub(lax.add(shape_term, log_linear_term), lax.log(scale))
  return where(logical_or(lax.gt(x, lax.add(loc, scale)),
                          lax.lt(x, loc)), -inf, log_probs)


@_wraps(osp_stats.beta.pdf, update_doc=False)
def pdf(x: ArrayLike, a: ArrayLike, b: ArrayLike,
        loc: ArrayLike = 0, scale: ArrayLike = 1) -> Array:
  return lax.exp(logpdf(x, a, b, loc, scale))
