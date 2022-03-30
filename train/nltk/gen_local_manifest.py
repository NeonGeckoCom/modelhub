import xdg.BaseDirectory
from pprint import pprint
from os.path import join
import os
from json_database import JsonStorageXDG
db = JsonStorageXDG("manifest", subfolder="ModelZoo")
db.clear()
path = join(xdg.BaseDirectory.xdg_cache_home, "ModelZoo", "nltk")

# TODO add sklearn folder etc.
for f in os.listdir(path):
    if f.endswith(".json"):
        continue
    db[f.replace(".pkl", "")] = f"https://github.com/NeonJarbas/modelhub/raw/models/models/nltk/{f}"

pprint(db)
db.store()
