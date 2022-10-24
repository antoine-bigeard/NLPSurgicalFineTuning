import yaml
import numpy as np
import pandas as pd


def read_yaml_config_file(path_config: str):
    with open(path_config) as conf:
        return yaml.load(conf, yaml.FullLoader)
