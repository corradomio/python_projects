from pprint import pprint
import mitsuba as mi

mi.set_variant('scalar_rgb')
print(mi.__version__)

data = mi.load_file('drop_scene_5.xml')
pprint(data)


