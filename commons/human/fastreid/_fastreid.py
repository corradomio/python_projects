from pathlib import Path

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F

from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#
# config files in "configs/..."
#
# FASTREID_MODEL_NAMES = [
#     # "Base-AGW",
#     # "Base-MGN",
#     # "Base-SBS",
#     # "Base-bagtricks",
#
#     "dukemtmc__AGW_R50",
#     "dukemtmc__AGW_R50-ibn",
#     "dukemtmc__AGW_R101-ibn",
#     "dukemtmc__AGW_S50",
#     "dukemtmc__bagtricks_R50",
#     "dukemtmc__bagtricks_R50-ibn",
#     "dukemtmc__bagtricks_R101-ibn",
#     "dukemtmc__bagtricks_S50",
#     "dukemtmc__mgn_R50-ibn",
#     "dukemtmc__sbs_R50",
#     "dukemtmc__sbs_R50-ibn",
#     "dukemtmc__sbs_R101-ibn",
#     "dukemtmc__sbs_S50",
#
#     "market1501__AGW_R50",
#     "market1501__AGW_R50-ibn",
#     "market1501__AGW_R101-ibn",
#     "market1501__AGW_S50",
#     "market1501__bagtricks_R50",
#     "market1501__bagtricks_R50-ibn",
#     "market1501__bagtricks_R101-ibn",
#     "market1501__bagtricks_S50",
#     "market1501__bagtricks_vit",
#     "market1501__mgn_R50-ibn",
#     "market1501__sbs_R50",
#     "market1501__sbs_R50-ibn",
#     "market1501__sbs_R101-ibn",
#     "market1501__sbs_S50",
#
#     # "msmt17: missing the datasets
#     "msmt17__AGW_R50",
#     "msmt17__AGW_R50-ibn",
#     "msmt17__AGW_R101-ibn",
#     "msmt17__AGW_S50",
#     "msmt17__bagtricks_R50",
#     "msmt17__bagtricks_R50-ibn",
#     "msmt17__bagtricks_R101-ibn",
#     "msmt17__bagtricks_S50",
#     "msmt17__mgn_R50-ibn",
#     "msmt17__sbs_R50",
#     "msmt17__sbs_R50-ibn",
#     "msmt17__sbs_R101-ibn",
#     "msmt17__sbs_S50",
# ]

FASTREID_WEIGHTS_URLS = {
    # "https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md
    # "market1501 Baselines
    "market1501/bagtricks_R50":       "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50.pth",
    "market1501/bagtricks_R50-ibn":   "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R50-ibn.pth",
    "market1501/bagtricks_S50":       "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_S50.pth",
    "market1501/bagtricks_R101-ibn":  "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_bot_R101-ibn.pth",
    "market1501/agw_R50":             "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50.pth",
    "market1501/agw_R50-ibn":         "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R50-ibn.pth",
    "market1501/agw_S50":             "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_S50.pth",
    "market1501/agw_R101-ibn":        "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_agw_R101-ibn.pth",
    "market1501/sbs_R50":             "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50.pth",
    "market1501/sbs_R50-ibn":         "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R50-ibn.pth",
    "market1501/sbs_S50":             "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_S50.pth",
    "market1501/sbs_R101-ibn":         "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_sbs_R101-ibn.pth",
    "market1501/mgn_R50-ibn":         "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/market_mgn_R50-ibn.pth",
    # "dukemtmc Baseline
    "dukemtmc/bagtricks_R50":         "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R50.pth",
    "dukemtmc/bagtricks_R50-ibn":     "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R50-ibn.pth",
    "dukemtmc/bagtricks_S50":         "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_S50.pth",
    "dukemtmc/bagtricks_R101-ibn":    "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_bot_R101-ibn.pth",
    "dukemtmc/agw_R50":               "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50.pth",
    "dukemtmc/agw_R50-ibn":           "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R50-ibn.pth",
    "dukemtmc/agw_S50":               "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_S50.pth",
    "dukemtmc/agw_R101-ibn":          "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_agw_R101-ibn.pth",
    "dukemtmc/sbs_R50":               "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50.pth",
    "dukemtmc/sbs_R50-ibn":           "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R50-ibn.pth",
    "dukemtmc/sbs_S50":               "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_S50.pth",
    "dukemtmc/sbs_R101-ibn":          "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_sbs_R101-ibn.pth",
    "dukemtmc/mgn_R50-ibn":           "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/duke_mgn_R50-ibn.pth",
    # "msmt17  Baseline
    "msmt17/bagtricks_R50":           "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R50.pth",
    "msmt17/bagtricks_R50-ibn":       "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R50-ibn.pth",
    "msmt17/bagtricks_S50":           "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_S50.pth",
    "msmt17/bagtricks_R101-ibn":      "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_bot_R101-ibn.pth",
    "msmt17/agw_R50":                 "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R50.pth",
    "msmt17/agw_R50-ibn":             "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R50-ibn.pth",
    "msmt17/agw_S50":                 "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_S50.pth",
    "msmt17/agw_R101-ibn":            "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_agw_R101-ibn.pth",
    "msmt17/sbs_R50":                 "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50.pth",
    "msmt17/sbs_R50-ibn":             "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R50-ibn.pth",
    "msmt17/sbs_S50":                 "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_S50.pth",
    "msmt17/sbs_R101-ibn":            "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_sbs_R101-ibn.pth",
    "msmt17/mgn_R50-ibn":             "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/msmt_mgn_R50-ibn.pth",
}


FASTREID_MODEL_NAMES = [
    name.replace("/", "__")
    for name in FASTREID_WEIGHTS_URLS.keys()
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

# where to save the models
FASTREID_WEIGHTS_ROOT = ".fastreid_weights"

# cache of predictors
FASTREID_MODELS = {}


def _get_cfg(cfg_name):
    cfg_file = Path(__file__).parent / f"configs/{cfg_name}.yml"
    cfg_file = str(cfg_file)
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    # cfg.freeze()
    return cfg


def _download_weights(weights_name, weights_path):
    assert weights_name in FASTREID_WEIGHTS_URLS
    url = FASTREID_WEIGHTS_URLS[weights_name]
    print(f"fastreid: downloading {weights_name} from {url} and saved in {weights_path}")
    response = requests.get(url)
    response.raise_for_status()
    with open(weights_path, "wb") as file:
        file.write(response.content)
    pass


def _set_weights_path(cfg_name, cfg):
    output_dir: str = cfg.OUTPUT_DIR
    if not output_dir.startswith("logs/"):
        assert output_dir.startswith("logs/")
    weights_name = output_dir[5:]
    # weights_name: <dataset>/<train_name>
    weights_path = Path(FASTREID_WEIGHTS_ROOT) / f"{weights_name}.pth"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    if not weights_path.exists():
        _download_weights(weights_name, weights_path)

    # safe the weights path in cfg.MODEL.WEIGHTS
    cfg.MODEL.WEIGHTS = str(weights_path)
    pass


def _get_model(cfg_name: str):
    global FASTREID_MODELS
    if cfg_name in FASTREID_MODELS:
        return FASTREID_MODELS[cfg_name]

    cfg = _get_cfg(cfg_name)
    _set_weights_path(cfg_name, cfg)
    predictor = DefaultPredictor(cfg)

    FASTREID_MODELS[cfg_name] = (cfg, predictor)
    return (cfg, predictor)
# end


# ---------------------------------------------------------------------------
# FastReID
# ---------------------------------------------------------------------------

class FastReID:

    @staticmethod
    def represent(image: str|Path|np.ndarray, model_name: str) -> np.ndarray:
        assert isinstance(image, (str, Path, np.ndarray))
        assert isinstance(model_name, str)

        if isinstance(image, (str, Path)):
            filename = image
            image = cv2.imread(filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cfg_name = model_name.replace("__", "/")
        cfg, predictor = _get_model(cfg_name)
        # h, w, c -> 256, 128, 3 /uint8
        image = cv2.resize(image, tuple(cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
        # b, c, h, w -> 1, 3, 256, 128
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
        features = predictor(image)

        features = F.normalize(features)
        features = features.cpu().data.numpy()

        # features: np.ndarray[1, len] -> np.ndarray[len]
        return features.reshape(-1)
    # end

    def __init__(self, model_name: str):
        assert isinstance(model_name, str)
        self._model_name = model_name

    def embedding(self, image: str | Path | np.ndarray):
        return FastReID.represent(image, self._model_name)
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
