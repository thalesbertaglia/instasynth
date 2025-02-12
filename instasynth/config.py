import json
import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict, List

from dotenv import find_dotenv, load_dotenv

# Logger setup
logger = logging.getLogger("instasynth")
logger.setLevel(logging.INFO)

# File handler, 5MB per log file, 3 log files max
file_handler = RotatingFileHandler(
    "instasynth.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logger.propagate = False

# API key
load_dotenv(find_dotenv())


@dataclass
class Config:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
    config_json_path: str = "../config.json"

    @classmethod
    def load_attributes(cls, config_json_path: str = None):
        if config_json_path:
            cls.config_json_path = config_json_path
        _json_config = cls.load_json_config()
        for k, v in _json_config.items():
            setattr(cls, k, v)
            if "PATH" in k:
                setattr(cls, k.replace("PATH", "FOLDER"), Path(v))
                Path(v).mkdir(parents=True, exist_ok=True)
        # setattr(cls, "FUNCTIONS", functions)
        # setattr(cls, "FUNCTION_CALL", function_call)

    @classmethod
    def load_json_config(cls) -> Dict:
        with open(cls.config_json_path, "r") as f:
            return json.load(f)
