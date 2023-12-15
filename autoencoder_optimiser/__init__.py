import json
import os

# Get the directory in which the current file (__init__.py) is located
package_directory = os.path.dirname(os.path.abspath(__file__))

default_config_path = os.path.join(package_directory, 'config.json')
custom_config_path = os.path.join(package_directory, 'config_custom.json')

CONFIG_PATH = custom_config_path if os.path.exists(custom_config_path) else default_config_path

config = None 

with open(CONFIG_PATH) as config_file:
    config = json.load(config_file)
