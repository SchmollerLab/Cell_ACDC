import os
import json

from ... import myutils

myutils.check_install_trackastra()

import trackastra

trackastra_folderpath = os.path.dirname(os.path.abspath(trackastra.__file__))
pretraned_json_filepath = os.path.join(
    trackastra_folderpath, 'model', 'pretrained.json'
)

def get_pretrained_model_names():
    with open(pretraned_json_filepath, encoding='utf-8') as file:
        json_data = json.load(file)
    
    return list(json_data.keys())