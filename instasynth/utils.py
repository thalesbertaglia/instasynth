import re
import json
from typing import List, Dict, Tuple, Union, Any
from pathlib import Path

import pandas as pd

from .config import Config


def load_json_config() -> Dict[str, Any]:
    with open("config.json", "r") as f:
        return json.load(f)


def read_prompt_json(prompt_name: str) -> Dict[str, Union[str, List[Dict[str, str]]]]:
    with open(
        Config.PROMPTS_FOLDER / f"{prompt_name}.json", "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def replace_params(text: str, parameters: Dict[str, str]) -> str:
    for param in parameters:
        text = text.replace(f"{{{param}}}", str(parameters[param]))
    return text


def get_filenames(experiment_identifier: str) -> Tuple[str, str]:
    experiment_results_path = Config.RESULTS_FOLDER / f"{experiment_identifier}"
    Path(experiment_results_path).mkdir(parents=True, exist_ok=True)

    experiment_results_filename = (
        f"{experiment_results_path}/df_{experiment_identifier}.csv"
    )
    setup_results_filename = (
        f"{experiment_results_path}/setup_{experiment_identifier}.json"
    )
    return experiment_results_filename, setup_results_filename


def process_response_content(content: str, split_pattern: str = "\n") -> pd.DataFrame:
    content = re.split(split_pattern, content.strip())
    df = pd.DataFrame(content)
    return df


def save_experiment_results(
    experiment_identifier: str,
    experiment_results: pd.DataFrame,
    setup_experiment: Dict[str, Union[str, Dict[str, str]]],
) -> None:
    experiment_results_filename, setup_results_filename = get_filenames(
        experiment_identifier
    )

    experiment_results.to_csv(experiment_results_filename, index=False)
    with open(setup_results_filename, "w", encoding="utf-8") as f:
        json.dump(setup_experiment, f, ensure_ascii=False)


def load_experiment_results(
    experiment_identifier: str,
) -> Tuple[pd.DataFrame, Dict[str, Union[str, Dict[str, str]]]]:
    experiment_results_filename, setup_results_filename = get_filenames(
        experiment_identifier
    )
    experiment_results = pd.read_csv(experiment_results_filename)
    with open(setup_results_filename, "r", encoding="utf-8") as f:
        setup_experiment = json.load(f)
    return experiment_results, setup_experiment
