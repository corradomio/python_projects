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


# def navigate(data, path, params=None, defval=None):
#     """
#         <path> ::= <step>/.../<step>
#         <step> ::= 'name' | 'name[<index>]' | 'name[{index}]'
#
#     :param map: a dicionary composed by list/dictionary
#     :param path: a xpath like expression
#     :param params: parameters used in xpath expression
#     :param defval: efalt value
#     :return:
#     """
#     def istep(s):
#         pos = s.find("[")
#         step = s[0:pos]
#         sel = s[pos+1:len(s)-1]
#         if sel.find("{") == -1:
#             index = int(sel)
#         else:
#             name = sel[1:len(sel)-1]
#             index = params[name]
#         return step, index
#
#     selected = data
#     steps = path.split("/")
#     for step in steps:
#         if selected is None:
#             break
#         if step == "" or step == ".":
#             continue
#         if step.find("[") == -1:
#             if step in selected:
#                 selected = selected[step]
#             else:
#                 selected = None
#         else:
#             step, index = istep(step)
#             if step in selected:
#                 selected = selected[step]
#                 selected = selected[index]
#             else:
#                 selected = None
#         # end
#     # end
#     if selected is None:
#         return defval
#     else:
#         return selected
# # end