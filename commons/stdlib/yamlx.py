#
# data is a structure composed by lists/tuples/sets & dictionaries
# path is an espression based on xpath
#
import yaml


class Navigate:

    @staticmethod
    def on(data):
        return Navigate(data)

    def __init__(self, data):
        self.data = data

    def __getattr__(self, item):        # data.item
        data = self.data
        if item not in data:
            return None
        selected = data[item]
        return self._wrap(selected)

    def __getitem__(self, item):        # data[item]
        data = self.data
        if type(data) not in [list, tuple, dict]:
            return None
        selected = data[item]
        return self._wrap(selected)

    def __len__(self):
        return len(self.data)

    def keys(self):
        return list(self.data.keys())

    def values(self):
        return list(self.data.values())

    def len(self):
        return len(self.data)

    def _wrap(self, selected):
        if type(selected) in [list, tuple, set, dict]:
            return Navigate(selected)
        else:
            return selected
# end


class YamlConfig(Navigate):

    def __init__(self, configfile):
        with open(configfile) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        super().__init__(data)
# end


# compatibility with 'json'

def load(fname: str, **kwargs):
    with open(fname, mode="r") as fp:
        return yaml.load(fp, yaml.Loader)

