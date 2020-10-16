import yaml
import os
import sys
import os.path as osp
import errno
import torch
import shutil
def get_config(config_path):
    with open(config_path, 'r') as stream:
        return yaml.load(stream)