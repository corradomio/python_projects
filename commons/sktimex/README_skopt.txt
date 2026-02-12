THIS is scikit-optimize     0.9.0
https://scikit-optimize.github.io/stable/

it is NOT scikit-optimize   0.10.1
https://github.com/holgern/scikit-optimize/blob/main/skopt/space/space.py


# [2026/02/11]
# The implementation of 'skopt.space.check_dimension(dimension)'
# check the conversion of [1,2] in Integer() (OLD implementation)
# and Categorical (NEW implementation).
# Because the old check is implemented by the function
#
# 'skopt.space.space._check_dimension_old'
#
# what we do is to replace the old implementation with the new
# one.
# The alternative approach was to reimplement the function
# 'skopt.space.check_dimension()', but is seemes it is not
# necessary.

from skopt.space.space import _check_dimension, _check_dimension_old

_check_dimension_old = _check_dimension
