#parse_config.py

import os
import json
from pathlib import Path

class Config:

    def __init__(self, config_file):
        self.config_file = config_file
        self.parse_config_file(self.config_file )

    
    def parse_config_file(self, config_file):
        with open(config_file) as cf:
            config = json.load(cf)
            self.__dict__.update(config)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

