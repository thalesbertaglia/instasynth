import re
import json
from typing import List, Dict, Tuple, Union, Any, Optional
from pathlib import Path

import pandas as pd

from .config import Config, logger

Config.load_attributes()


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


def get_pathnames(experiment_identifier: str) -> Tuple[str, str]:
    experiment_results_path = Config.RESULTS_FOLDER / f"{experiment_identifier}"
    Path(experiment_results_path).mkdir(parents=True, exist_ok=True)

    return experiment_results_path


def process_response_content(content: str) -> pd.DataFrame:
    try:
        return pd.DataFrame(
            json.loads(fix_json_string(content))["posts"], columns=["caption"]
        )
    except Exception as e:
        logger.error(f"Failed to parse the JSON response: {e}")
        raise e


def fix_json_string(json_str: str) -> str:
    """Format a JSON string to correct common mistakes."""
    # Normalize line breaks and strip whitespace
    normalized_lines = [
        line.strip()
        for line in json_str.replace("\r\n", "\n").splitlines()
        if line.strip()
    ]
    normalized_str = "\n".join(normalized_lines).replace("\\\n", "\n")
    # Ensure proper comma placements between consecutive quotes (between posts)
    with_commas = re.sub(r'"\s*\n"', r'",\n"', normalized_str)
    # Remove trailing commas in lists or objects
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", with_commas)
    # Remove unnecessary whitespace characters
    whitespace_cleaned = no_trailing_commas.replace("\t", " ").replace("\n", " ")

    return whitespace_cleaned


def format_examples(examples: list, delimiter: str) -> str:
    def format_example(example):
        fixed_example = (
            example.replace('"', " ")
            .replace("'", " ")
            .replace("“", " ")
            .replace("’", " ")
            .replace("‘", " ")
            .replace("⠀", "")
            .replace(delimiter, " ")
        )
        fixed_example = re.sub(r"\s+", " ", fixed_example)
        return fixed_example

    """Format examples for display."""
    return "".join([f"{delimiter}{format_example(example)}\n" for example in examples])


def format_text(text: str) -> str:
    """Format text for embedding generation"""
    fixed_text = text.lower()
    fixed_text = fixed_text.replace("\\n", " ").replace("\\t", " ")
    fixed_text = re.sub(r"\s+", " ", fixed_text)
    return fixed_text


def sample_examples(
    df: pd.DataFrame,
    n_examples: int,
    is_sponsored: bool = False,
    sponsorship_column: str = "has_disclosures",
    random_seed: Optional[int] = None,
    delimiter: str = "####",
) -> Tuple[List[str], List[str]]:
    examples = df.query(f"{sponsorship_column} == {is_sponsored}").sample(
        n_examples, random_state=random_seed
    )[["shortcode", "caption"]]

    return (
        format_examples(examples["caption"].tolist(), delimiter),
        examples["shortcode"],
    )
