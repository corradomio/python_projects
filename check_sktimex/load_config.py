from stdlib import jsonx


def _extract_model_selector(config: dict) -> tuple[str, dict, dict]:
    keys = config.keys()
    for k in keys:
        if k.startswith("$"):
            ms_name = k[1:]
            ms_config = config[k]
            del config[k]
            return ms_name, ms_config, config
    raise ValueError("No model selector found")
# end

def _extract_param_grid(config: dict, ms_name) -> tuple[dict, dict, dict]:
    keys = list(config.keys())
    param_grid = {}
    ms_override = {}
    for k in keys:
        if not k.startswith("$"):
            continue

        if k == ms_name:
            ms_override = config[k]
            del config[k]
            continue

        pname = k[1:]
        pvalues = config[k]

        # 1) remove '$<name>'
        del config[k]
        # 2) add '<name>'
        config[pname] = pvalues[0]
        # 3) fill 'param_grid'
        param_grid[pname] = pvalues
    # end
    return param_grid, config, ms_override

def _compose_model_selection_model_name(msname: str, mname: str) -> str:
    p = msname.split(".")
    return f"{p[0]}.{mname}.{p[1]}"


def load_model_selection_config(config_file: str) -> dict:
    config = jsonx.load(config_file)
    conposed_config = {}

    # 1) extract the configuration of the model selector (ms) name and config
    ms_name, ms_config, config = _extract_model_selector(config)

    # 2) scan the models and extract the parameters to put in the 'param_grid'
    for m_name in config:
        param_grid, m_config0, ms_override = _extract_param_grid(config[m_name], ms_name)

        # if 'param_grid' is empty, do nothing
        if len(param_grid) == 0:
            continue

        # 2.1) compose the model selection config
        msm_name = _compose_model_selection_model_name(ms_name, m_name)

        msm_config = {}|ms_config|ms_override
        msm_config["forecaster"] = m_config0
        msm_config["param_grid"] = param_grid

        # 2.2) populate the global configuration
        conposed_config[msm_name] = msm_config
    # end

    return conposed_config
# end
