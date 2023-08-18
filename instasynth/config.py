import os
import sys
import json
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from dotenv import load_dotenv, find_dotenv

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


load_dotenv(find_dotenv())


def load_json_config():
    with open("../config.json", "r") as f:
        return json.load(f)


class Config:
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    @classmethod
    def load_attributes(cls):
        _json_config = load_json_config()
        for k, v in _json_config.items():
            setattr(cls, k, v)
        setattr(cls, "PROMPTS_FOLDER", Path(cls.PROMPTS_PATH))
        setattr(cls, "RESULTS_FOLDER", Path(cls.RESULTS_PATH))


Config.load_attributes()
