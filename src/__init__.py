import json
import os

CONFIG_PATH = "../config.json"

if(os.path.exists("../config_custom.json")):
    CONFIG_PATH = "../config_custom.json"


config = None 

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)



