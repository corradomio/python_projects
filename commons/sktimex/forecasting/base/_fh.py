# from pandas import Index
#
# import sktime.forecasting.base as skt
#
#
# class ForecastingHorizonExt(skt.ForecastingHorizon):
#     """
#     Extends sktime ForecastingHorizon in the following way:
#     """
#
#     def __init__(self, values=None, is_relative=None, freq=None):
#         """
#         Enhanced version of Sktime ForecastingHorizon
#
#         :param values: if it is an integer, it is converted in the list [1,...,value]
#
#         :param is_relative:
#         :param freq:
#         """
#         if isinstance(values, int):
#             super().__init__(list(range(1, values+1)), freq=freq, is_relative=True)
#         elif isinstance(values, type(range())):
#             super().__init__(list(values), freq=freq, is_relative=True)
#         elif isinstance(values, Index):
#             super().__init__(values, freq=freq, is_relative=False)
#         else:
#             super().__init__(values, is_relative=is_relative, freq=freq)
# # end
