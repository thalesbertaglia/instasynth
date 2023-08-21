import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass, field

import openai
import pandas as pd

from .config import Config, logger
from . import utils


@dataclass
class DataGenerator:
    model_parameters: Dict[str, Union[str, float, int, List[Dict[str, str]]]] = field(
        default_factory=lambda: {
            "model": Config.MODEL,
            "functions": Config.FUNCTIONS,
            "function_call": Config.FUNCTION_CALL,
            "temperature": Config.TEMPERATURE,
            "frequency_penalty": Config.FREQUENCY_PENALTY,
            "presence_penalty": Config.PRESENCE_PENALTY,
            "max_tokens": Config.MAX_TOKENS,
        }
    )

    @staticmethod
    def transform_message(
        message: Dict[str, str], parameters: Dict[str, str]
    ) -> Dict[str, str]:
        message_type, message_content = next(iter(message.items()))
        return {
            "role": message_type,
            "content": utils.replace_params(message_content, parameters),
        }

    def create_prompt_messages(
        self, messages: List[Dict[str, str]], parameters: Dict[str, str]
    ) -> List[Dict[str, str]]:
        return [self.transform_message(message, parameters) for message in messages]

    def get_completion_from_messages(
        self, messages: List[Dict[str, str]]
    ) -> Union[str, Tuple[Dict, str, Dict[str, Union[int, str]]]]:
        try:
            response = openai.ChatCompletion.create(
                messages=messages, **self.model_parameters
            )
            arguments = response["choices"][0]["message"]["function_call"]["arguments"]
            return response, arguments, dict(response["usage"])
        except Exception as e:
            logger.error(e)
            return "API_ERROR"

    def generate_data(
        self,
        prompt_name: str,
        parameters: Dict[str, str] = None,
        chatgpt_parameters: Dict[str, str] = None,
    ) -> Dict:
        if parameters is None:
            parameters = {}
        if chatgpt_parameters is None:
            chatgpt_parameters = {}

        prompt_json = utils.read_prompt_json(prompt_name)
        messages = self.create_prompt_messages(prompt_json["messages"], parameters)
        logger.info("Sending messages to OpenAI API...")
        response = self.get_completion_from_messages(messages, **chatgpt_parameters)
        logger.info("Received response from OpenAI API!")

        return response
