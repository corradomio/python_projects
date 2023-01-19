class ClassicDict:

    @staticmethod
    def open(*args, ktype=None, vtype=None, **kwargs):
        return ClassicDict()

    def select(self, name: str, ktype=None, vtype=None):
        return dict()

    def __init__(self, *args, **kwargs):
        self.dict = dict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __setitem__(self, key, value):
        self.dict[key] = value

    def __getitem__(self, key):
        return self.dict[key]

    def __contains__(self, key):
        return key in self.dict

    def __iter__(self):
        return self.dict.__iter__()

    def __len__(self):
        return self.dict.__len__()

# end
