from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

from litellm import completion

from . import utils
from .config import Config

Config.load_attributes()


@dataclass
class DataGenerator:
    model_parameters: Dict[str, Union[str, float, int, List[Dict[str, str]]]] = field(
        default_factory=lambda: {
            "model": Config.MODEL,
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
            response = completion(
                messages=messages, drop_params=True, **self.model_parameters
            ).json()
            arguments = response["choices"][0]["message"]["content"]
            # arguments = response["choices"][0]["message"]["function_call"]["arguments"]
            return response, arguments
        except Exception as e:
            raise e

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
        else:
            self.model_parameters.update(chatgpt_parameters)

        prompt_json = utils.read_prompt_json(prompt_name)
        messages = self.create_prompt_messages(prompt_json["messages"], parameters)
        response = self.get_completion_from_messages(messages)

        return response, messages
