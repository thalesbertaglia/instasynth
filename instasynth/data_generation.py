import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Union

import openai
import pandas as pd

from .config import Config, logger
from .utils import (
    read_prompt_json,
    replace_params,
    get_filenames,
    process_response_content,
    save_experiment_results,
    load_experiment_results,
)


def create_openai_messages(
    messages: List[Dict[str, str]], parameters: Dict[str, str]
) -> List[Dict[str, str]]:
    prompt_messages = []
    for message in messages:
        message_type = list(message.keys())[0]
        message_content = replace_params(list(message.values())[0], parameters)
        prompt_messages.append({"role": message_type, "content": message_content})
    return prompt_messages


def get_completion_from_messages(
    messages: List[Dict[str, str]],
    model: str = Config.MODEL,
    temperature: float = Config.TEMPERATURE,
    frequency_penalty: float = Config.FREQUENCY_PENALTY,
    presence_penalty: float = Config.PRESENCE_PENALTY,
    max_tokens: int = Config.MAX_TOKENS,
) -> Union[str, Tuple[Dict, str, Dict[str, Union[int, str]]]]:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            # n=2,
        )

        return response, response.choices[0].message["content"], dict(response["usage"])
    except Exception as e:
        logger.error(e)
        return "API_ERROR"


def create_experiment(
    experiment_identifier: str, prompt_name: str, parameters: Dict[str, str]
) -> Tuple[pd.DataFrame, Dict[str, Union[str, Dict[str, str]]]]:
    logger.info(
        f"Running experiment {experiment_identifier} with prompt {prompt_name} and parameters {parameters}"
    )
    experiment_results_filename, _ = get_filenames(experiment_identifier)
    prompt_json = read_prompt_json(prompt_name)
    messages = create_openai_messages(prompt_json["messages"], parameters)

    logger.info("Sending messages to OpenAI API...")
    full_response, response_content, response_usage = get_completion_from_messages(
        messages
    )
    logger.info("Received response from OpenAI API!")

    logger.info("Processing response...")
    experiment_results = process_response_content(response_content)
    setup_experiment = {
        "experiment_identifier": experiment_identifier,
        "prompt_name": prompt_name,
        "df_results": experiment_results_filename,
        "parameters": parameters,
        "response_usage": dict(response_usage),
    }
    logger.info("Storing results...")
    save_experiment_results(experiment_identifier, experiment_results, setup_experiment)
    logger.info("Experiment complete!")
    return experiment_results, setup_experiment


def create_setup_dict(
    experiment_identifier: str, prompt_name: str, parameters: Dict[str, str]
) -> Dict[str, Union[str, Dict[str, str]]]:
    experiment_results_filename, _ = get_filenames(experiment_identifier)
    return {
        "experiment_identifier": experiment_identifier,
        "prompt_name": prompt_name,
        "experiment_results_filename": experiment_results_filename,
        "parameters": parameters,
    }
