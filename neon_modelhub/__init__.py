import os
from os.path import join, isfile

import joblib
import requests
import xdg.BaseDirectory

from neon_modelhub.manifest import MANIFEST

MODELS_DIR = join(xdg.BaseDirectory.xdg_cache_home, "ModelZoo")


def load_model(model_id):
    # model_id is {framework}_{dataset}_{algorithm}_{task}
    subfolder = model_id.split("_")[0]
    models_dir = join(MODELS_DIR, subfolder)
    model_path = join(models_dir, model_id)
    if not model_path.endswith(".pkl"):
        model_path += ".pkl"
    if not isfile(model_path):
        download_model(model_id)
    return joblib.load(model_path)


def download_model(model_id):
    # model_id is {framework}_{dataset}_{algorithm}_{task}
    url = MANIFEST[model_id]
    subfolder = model_id.split("_")[0]
    models_dir = join(MODELS_DIR, subfolder)
    os.makedirs(models_dir, exist_ok=True)
    model_path = join(models_dir, model_id + ".pkl")
    r = requests.get(url)
    with open(model_path, "wb") as f:
        f.write(r.content)
