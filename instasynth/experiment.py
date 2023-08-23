import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
from enum import Enum
from pathlib import Path

import pandas as pd

from .config import Config, logger
from .data_generation import DataGenerator
from . import utils


class SavingManager:
    def __init__(self, experiment_identifier: str):
        self.experiment_results_filename = utils.get_pathnames(experiment_identifier)
        Path(self.experiment_results_filename / "iterations").mkdir(
            parents=True, exist_ok=True
        )
        Path(self.experiment_results_filename / "prompt_examples").mkdir(
            parents=True, exist_ok=True
        )
        Path(self.experiment_results_filename / "errors").mkdir(
            parents=True, exist_ok=True
        )

    def save_iteration_results(
        self,
        full_response: Dict,
        messages: List[Dict[str, str]],
        generated_posts: pd.DataFrame,
        sponsorship: str,
        iteration_state: int,
        examples_shortcodes: List[str] = None,
    ):
        # Posts per iteration
        logger.info(f"{len(generated_posts)} posts generated.")
        generated_posts.to_pickle(
            f"{self.experiment_results_filename}/iterations/posts_{sponsorship}_{iteration_state}.pkl"
        )
        # Full response
        with open(
            f"{self.experiment_results_filename}/iterations/{sponsorship}_{iteration_state}.json",
            "w",
        ) as f:
            json.dump(full_response, f, indent=4, ensure_ascii=False)
        with open(
            f"{self.experiment_results_filename}/iterations/msg_{sponsorship}_{iteration_state}.json",
            "w",
        ) as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
        # Example posts, if any
        if examples_shortcodes is not None:
            examples_shortcodes.to_pickle(
                f"{self.experiment_results_filename}/prompt_examples/{sponsorship}_{iteration_state}.pkl"
            )

    def save_error_post(self, error_post: str, sponsorship: str, iteration_state: int):
        with open(
            f"{self.experiment_results_filename}/errors/{sponsorship}_{iteration_state}.txt",
            "w",
        ) as f:
            f.write(error_post)

    def save_final_results(
        self, results: Dict[str, List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        dfs = []
        for sponsorship in ["sponsored", "nonsponsored"]:
            df = pd.concat(results[sponsorship])
            df["sponsorship"] = sponsorship
            dfs.append(df)

        df = pd.concat(dfs)
        df.to_pickle(f"{self.experiment_results_filename}/final_df.pkl")
        return df

    def save_experiment_setup(self, experiment_data: Dict[str, Any]):
        """Save the experiment setup to a JSON file, ignoring fields marked with 'ignore' in metadata."""

        with open(
            f"{self.experiment_results_filename}/EXPERIMENT_SETUP.json",
            "w",
            encoding="utf8",
        ) as f:
            json.dump(experiment_data, f, indent=4, ensure_ascii=False)


class Sponsorship(Enum):
    SPONSORED = "sponsored"
    NON_SPONSORED = "nonsponsored"


@dataclass
class Experiment:
    experiment_identifier: str
    sponsored_prompt_name: str
    nonsponsored_prompt_name: str
    sponsored_count: int
    nonsponsored_count: int
    captions_per_prompt: int = Config.CAPTIONS_PER_PROMPT
    number_of_examples: int = 0
    sample_examples: bool = False
    data_generator: DataGenerator = field(default_factory=DataGenerator)
    # Additional parameters to be passed to the prompt (e.g. fixed examples)
    parameters_template: Dict[str, str] = field(default_factory=dict)
    chatgpt_parameters: Dict[str, str] = field(default_factory=dict)
    results: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {"sponsored": [], "nonsponsored": []},
        metadata={"ignore": True},
    )
    _iteration_state: int = 0
    _n_requests: int = 0

    def __post_init__(self):
        self.saving_manager: SavingManager = SavingManager(self.experiment_identifier)
        self.parameters_template["number_of_captions"] = self.captions_per_prompt

    def sponsorship_type(self, is_sponsored: bool) -> str:
        return (
            Sponsorship.SPONSORED.value
            if is_sponsored
            else Sponsorship.NON_SPONSORED.value
        )

    def run_single(self, is_sponsored: bool, df_sample: pd.DataFrame = None):
        parameters = self.parameters_template.copy()
        sponsorship = self.sponsorship_type(is_sponsored)
        prompt_name = (
            self.sponsored_prompt_name
            if is_sponsored
            else self.nonsponsored_prompt_name
        )
        if self.sample_examples:
            examples, examples_shortcodes = utils.sample_examples(
                df=df_sample,
                n_examples=self.number_of_examples,
                is_sponsored=is_sponsored,
            )
            parameters[
                "examples"
            ] = f"<EXAMPLES> {examples} </EXAMPLES>\n\n Now I will give you instructions: "

        (full_response, response_content), messages = self.data_generator.generate_data(
            prompt_name=prompt_name,
            parameters=parameters,
            chatgpt_parameters=self.chatgpt_parameters,
        )
        # If formatting the response fails, ignore this run and continue
        try:
            generated_posts = utils.process_response_content(content=response_content)
            if self.sample_examples:
                generated_posts["examples_shortcodes"] = [
                    examples_shortcodes.tolist()
                ] * len(generated_posts)
            logger.info(
                f"Storing results for iteration {sponsorship}_{self._iteration_state}..."
            )
            # Updating the result dict
            if generated_posts is not None:
                self.results[sponsorship].append(generated_posts)
                examples_shortcodes = (
                    examples_shortcodes if self.sample_examples else None
                )
                self.saving_manager.save_iteration_results(
                    full_response=full_response,
                    messages=messages,
                    generated_posts=generated_posts,
                    sponsorship=sponsorship,
                    iteration_state=self._iteration_state,
                    examples_shortcodes=examples_shortcodes,
                )

        except Exception as e:
            logger.error(f"Failed to process response ({e}). Continuing...")
            self.saving_manager.save_error_post(
                response_content, sponsorship, self._iteration_state
            )

    def run(self, df_sample: pd.DataFrame = None):
        logger.info(
            f"Running experiment {self.experiment_identifier}. Generating {self.sponsored_count} sponsored posts and {self.nonsponsored_count} non-sponsored posts."
        )
        # Saving early in case the experiment fails
        self.save_experiment_setup()
        # Loop over the two types of posts
        for is_sponsored in [True, False]:
            self._iteration_state = 0
            sponsorship = self.sponsorship_type(is_sponsored)
            sponsorship_count = (
                self.sponsored_count if is_sponsored else self.nonsponsored_count
            )
            # Loop over the number of posts to generate
            generated_posts_count = 0
            while generated_posts_count < sponsorship_count:
                self._n_requests += 1
                self._iteration_state += 1
                logger.info(
                    f"Generating {sponsorship} posts, iteration {self._iteration_state}. {generated_posts_count}/{sponsorship_count} generated so far."
                )
                self.run_single(is_sponsored=is_sponsored, df_sample=df_sample)
                generated_posts_count = sum((len(k) for k in self.results[sponsorship]))
        logger.info("Finished generating all posts!")
        # Saving the setup again to ensure parameters are up to date
        self.save_experiment_setup()
        df = self.saving_manager.save_final_results(results=self.results)
        return df

    def save_experiment_setup(self):
        experiment_dict = asdict(self)

        # Removing fields marked with 'ignore' in metadata
        for field_name, field_value in self.__dataclass_fields__.items():
            if field_value.metadata.get("ignore"):
                experiment_dict.pop(field_name, None)
        self.saving_manager.save_experiment_setup(experiment_dict)
