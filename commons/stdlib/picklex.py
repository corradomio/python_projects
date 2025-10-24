import pickle


def dump(obj, file):
    with open(file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(file):
    with open(file, 'rb') as handle:
        obj = pickle.load(handle)
    return obj
