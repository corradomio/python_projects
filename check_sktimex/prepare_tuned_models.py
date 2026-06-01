from collections import defaultdict

from stdlib import jsonx
from common import *

#
# Create the JSON configuration files with the list of models & timeseres to tune
#


def main():
    data = csvx.load("stats/models_plain_statistics.csv", skiprows=1)
    n = len(data)
    auto_models = defaultdict(lambda: {})

    for rec in data:
        lib, name, cat, mean, quality = rec

        if "36" in cat:
            continue

        if mean < MSE_BAD:
            continue

        # if lib not in auto_models:
        #     auto_models[lib] = {}

        lib_auto_models = auto_models[lib]

        key = f"{lib}.{name}"
        if not key in lib_auto_models:
            lib_auto_models[key] = {
                "class": None,
                "+datasets": []
            }

        auto_model = lib_auto_models[key]

        auto_model["+datasets"].append(cat)
    # end

    for lib in auto_models:
        auto_file = f"config_ext/auto_{lib}_models_ext.json"
        auto_config = auto_models[lib]
        jsonx.dump(auto_config, auto_file)

    pass



if __name__ == "__main__":
    main()