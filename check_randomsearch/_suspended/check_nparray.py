import numpy as np
import pandas as pd
import ctypes


class NPSingleton:

    def __init__(self, value):
        self.value = value
        # self.__array_interface__ = None
        # self.__array_priority__ = None
        # self.__array_struct__ = None

    def __getattr__(self, item):
        pass

    def __len__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getitem__(self, item):
        pass

    def __getattribute__(self, item):
        # _array_struct__
        return self
        # return None

    def __class_getitem__(cls, item):
        pass

    def __get__(self, instance, owner):
        pass

    # def __array__(self, dtype=None):
    #     return np.array([self.value])
    #
    # def __array_finalyze__(self, obj):
    #     pass
    #
    # def __array_function__(self, func, types, args, kwargs):
    #     pass
    #
    # def __array_prepare__(self, array, context):
    #     pass
    #
    # def __array_ufunc__(self, ufunc, method, inputs, kwargs):
    #     pass
    #
    # def __array_wrap__(self, array, context):
    #     pass


def main():
    df = pd.DataFrame(data=[[11, 12], [21, 22]], columns=['c1', 'c2'])

    l = [df]
    # a = np.array(l)
    # print(a.__array_interface__)
    # print(a.__array_priority__)
    # print(a.__array_struct__)
    # oid = a.__array_interface__['data'][0]
    # obj = ctypes.cast(oid, ctypes.py_object).value
    # print(obj)

    # df = NPSingleton(df)
    #
    item = np.array(df).item()
    # print(type(item))
    pass


if __name__ == "__main__":
    main()
