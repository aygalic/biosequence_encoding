import json

CONFIG_PATH = "../config.json"

config = None 

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)



