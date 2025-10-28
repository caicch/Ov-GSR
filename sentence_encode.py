import os
import argparse
from tqdm import tqdm

import torch

from clip.clip import load, tokenize
import json

f = open('/mnt/data/SWIG/SWiG_jsons/train_with_obj_cap_refined.json',  mode = 'r')
data = json.load(f)

f = open('/mnt/data/SWIG/SWiG_jsons/imsitu_space.json',  mode = 'r')
data_roles = json.load(f)


for image, dic in tqdm(data.items(), desc="Extracting data"):
    cap_array = []
    cap_array.append(dic['verb'])
    im_path = "/mnt/data/SWIG/images_512/"+image
    for role, id in dic['frames'][0].items():
        if id:
            if id in data_roles['nouns']:
                cap_array.append(data_roles['nouns'][id]['gloss'][0])