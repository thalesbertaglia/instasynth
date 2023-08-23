import re
import json
from typing import List, Dict, Tuple, Union, Any, Optional
from pathlib import Path

import pandas as pd

from .config import Config, logger


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
            json.loads(format_json_string(content))["posts"], columns=["caption"]
        )
    except Exception as e:
        logger.error(f"Failed to parse the JSON response: {e}")
        raise e


def format_json_string(json_str: str) -> str:
    """Format a JSON string to correct common mistakes."""
    # Simplify whitespace, but preserve structure
    json_str = "\n".join(line.strip() for line in json_str.splitlines() if line.strip())
    # Replace \r\n with \n for consistency
    json_str = json_str.replace("\r\n", "\n")
    # Remove forward slashes before line breaks
    json_str = json_str.replace("\\\n", "\n")
    # Add missing commas between consecutive quotes
    json_str = re.sub(r'("\s+)\n(")', r"\1,\n\2", json_str)
    try:
        _ = json.loads(json_str)
    except json.JSONDecodeError:
        # Combine split posts into one line
        json_str = re.sub(r'(?<=,)\n"', ' "', json_str)
        # Add missing commas between posts
        json_str = re.sub(r'("\n)\s*"', r"\1,", json_str)
        # Remove multiple line breaks (keeping a single line break)
        json_str = re.sub(r"\n+", "\n", json_str)
        # Remove consecutive commas
        json_str = re.sub(r",\s*,", ",", json_str)
        # Remove trailing commas
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

        return json_str

    return json_str


def format_examples(examples: list) -> str:
    """Format examples for display."""
    return "".join(
        [f"<POST#{index}> {example}\n" for index, example in enumerate(examples)]
    )


def sample_examples(
    df: pd.DataFrame,
    n_examples: int,
    is_sponsored: bool = False,
    sponsorship_column: str = "has_disclosures",
    random_seed: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    examples = df.query(f"{sponsorship_column} == {is_sponsored}").sample(
        n_examples, random_state=random_seed
    )[["shortcode", "caption"]]

    return format_examples(examples["caption"].tolist()), examples["shortcode"]
