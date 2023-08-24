import re
import json
from typing import Dict, List, Tuple, ClassVar, Set, Union, Optional
from dataclasses import dataclass, field
from pathlib import Path

import textstat
import emoji
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


@dataclass
class ExperimentLoader:
    experiment_path: Path
    _errors_path: Path = field(init=False)
    _iterations_path: Path = field(init=False)
    _n_requests: int = field(init=False, default=None)
    _n_errors: int = field(init=False, default=None)
    _captions_per_request: np.ndarray = field(init=False, default=None)

    ERRORS_DIR: ClassVar[str] = "errors"
    ITERATIONS_DIR: ClassVar[str] = "iterations"
    SETUP_FILE: ClassVar[str] = "EXPERIMENT_SETUP.json"
    # Df with the df with all results
    DF_FILE: ClassVar[str] = "final_df.pkl"

    def __post_init__(self):
        self._errors_path = self.experiment_path / self.ERRORS_DIR
        self._iterations_path = self.experiment_path / self.ITERATIONS_DIR

    def load_n_requests(self) -> int:
        if self._n_requests is None:
            with (self.experiment_path / self.SETUP_FILE).open("r") as f:
                self._n_requests = json.load(f)["_n_requests"]
        return self._n_requests

    @property
    def n_errors(self) -> int:
        if self._n_errors is None:
            self._n_errors = len(list(self._errors_path.iterdir()))
        return self._n_errors

    @property
    def captions_per_request(self) -> np.ndarray:
        if self._captions_per_request is None:
            self._captions_per_request = np.array(
                [len(pd.read_pickle(f)) for f in self._iterations_path.glob("*.pkl")]
            )
        return self._captions_per_request

    def extract_metrics(self) -> Dict[str, Union[int, float]]:
        n_requests = self.load_n_requests()
        n_errors = self.n_errors
        captions_per_request = self.captions_per_request

        # Calculate metrics
        metrics = {
            "number_of_requests": n_requests,
            "number_of_errors": n_errors,
            "avg_captions_per_request": captions_per_request.mean(),
            "std_captions_per_request": captions_per_request.std(),
            "max_captions_per_request": captions_per_request.max(),
            "min_captions_per_request": captions_per_request.min(),
            "error_rate": n_errors / n_requests
            if n_requests != 0
            else 0,  # Proportion of requests that resulted in errors
            "success_rate": 1 - (n_errors / n_requests)
            if n_requests != 0
            else 0,  # Proportion of successful requests
            "total_captions": captions_per_request.sum(),
        }

        return metrics

    def get_experiment_final_df(self) -> pd.DataFrame:
        return pd.read_pickle(self.experiment_path / self.DF_FILE)


@dataclass
class TextAnalyser:
    data: pd.DataFrame
    _tokenized_captions: pd.Series = None
    _stopwords: Set[str] = field(default_factory=set)
    __HASHTAG_PATTERN: ClassVar[re.Pattern] = re.compile(r"#(\w+)")
    __USER_TAG_PATTERN: ClassVar[re.Pattern] = re.compile(r"@(\w+)")

    def __post_init__(self):
        if "caption" not in self.data.columns:
            raise ValueError("DataFrame must contain 'caption' column.")

        self._stopwords = set(stopwords.words("english"))
        self.tokenizer = TweetTokenizer(preserve_case=False)

    @property
    def tokenized_captions(self) -> pd.Series:
        if self._tokenized_captions is None:
            self._tokenized_captions = self.data["caption"].apply(
                self.tokenizer.tokenize
            )
        return self._tokenized_captions

    def _extract_from_pattern(
        self, tokens: List[str], pattern: re.Pattern
    ) -> List[str]:
        return [
            match.group(1)
            for token in tokens
            for match in [pattern.search(token)]
            if match
        ]

    def _analyse_patterns(self, pattern: re.Pattern) -> Tuple[float, float, pd.Series]:
        extracted = self.tokenized_captions.apply(
            lambda x: self._extract_from_pattern(x, pattern)
        )
        avg_per_post = extracted.str.len().mean()
        std = extracted.str.len().std()
        total_extracted = pd.Series([item for sublist in extracted for item in sublist])
        return avg_per_post, std, total_extracted

    def analyse_data(self) -> pd.DataFrame:
        metrics = self._basic_metrics()
        metrics.update(self._pattern_based_metrics())
        metrics.update(self._text_complexity_metrics())
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
        return metrics_df

    def _basic_metrics(self) -> Dict[str, float]:
        return {
            "avg_caption_length": self.tokenized_captions.apply(len).mean(),
            "vocabulary_size": pd.Series(
                token
                for tokens in self.tokenized_captions
                for token in tokens
                if token not in self._stopwords
            ).nunique(),
            "avg_emojis_per_post": self.data["caption"].apply(emoji.emoji_count).mean(),
            "std_emojis_per_post": self.data["caption"].apply(emoji.emoji_count).std(),
            "n_unique_emojis": len(
                set(
                    emoji_char
                    for caption in self.data["caption"]
                    for emoji_char in emoji.distinct_emoji_list(caption)
                )
            ),
        }

    def _pattern_based_metrics(self) -> Dict[str, Union[float, int]]:
        metrics = {}
        features = {
            "hashtags": self.__HASHTAG_PATTERN,
            "user_tags": self.__USER_TAG_PATTERN,
        }

        for feature_name, pattern in features.items():
            avg_per_post, std, total_extracted = self._analyse_patterns(pattern)
            metrics[f"avg_{feature_name}_per_post"] = avg_per_post
            metrics[f"std_{feature_name}_per_post"] = std
            metrics[f"total_{feature_name}"] = total_extracted.shape[0]
            metrics[f"n_unique_{feature_name}"] = total_extracted.nunique()

        return metrics

    def _text_complexity_metrics(self) -> Dict[str, float]:
        fk_grade_levels = self.data["caption"].apply(textstat.flesch_kincaid_grade)
        dalle_readability = self.data["caption"].apply(
            textstat.dale_chall_readability_score
        )
        return {
            "avg_fk_grade_level": fk_grade_levels.mean(),
            "std_fk_grade_level": fk_grade_levels.std(),
            "avg_dalle_readability": dalle_readability.mean(),
            "std_dalle_readability": dalle_readability.std(),
        }


@dataclass
class ClassificationAnalyser:
    identifier: str = ""
    data: pd.DataFrame
    evaluation_data: pd.DataFrame
    text_column: str = "caption"
    target_column: str = "sponsorship"

    __VECTORIZER: ClassVar[TfidfVectorizer] = TfidfVectorizer()
    __MODEL: ClassVar[LogisticRegression] = LogisticRegression()

    def __post_init__(self):
        if (
            self.text_column not in self.data.columns
            or self.target_column not in self.data.columns
        ):
            raise ValueError(
                "DataFrame must contain the source (text) and target column."
            )
        self.__VECTORIZER.fit(self.data[self.text_column])
        self.data[self.target_column] = self.data[self.target_column].map(
            {"sponsored": 1, "nonsponsored": 0}
        )

    def _preprocess(self, data: pd.DataFrame) -> tuple:
        X = self.__VECTORIZER.transform(data[self.text_column])
        y = data[self.target_column].values
        return X, y

    def train(self, X_train, y_train) -> None:
        self.__MODEL.fit(X_train, y_train)

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        predictions = self.__MODEL.predict(X_test)
        metrics = {
            # f"{self.identifier}_accuracy": accuracy_score(y_test, predictions),
            f"{self.identifier}_precision": precision_score(y_test, predictions),
            f"{self.identifier}_recall": recall_score(y_test, predictions),
            f"{self.identifier}_f1_score": f1_score(y_test, predictions),
        }
        return metrics

    def ad_detection(self) -> Dict[str, float]:
        X_train, y_train = self._preprocess(self.data)
        X_test, y_test = self._preprocess(self.evaluation_data)
        self.train(X_train, y_train)
        return self.evaluate(X_test, y_test)


@dataclass
class ExperimentEvaluator:
    experiment_paths: List[Path]
    real_dataset: Optional[pd.DataFrame] = None

    _experiment_metrics: Dict[str, Dict[str, Union[int, float]]] = field(
        default_factory=dict, init=False
    )
    _real_dataset_metrics: Dict[str, Union[int, float]] = field(
        default_factory=dict, init=False
    )

    def _load_and_analyse_experiment(self, path: Path) -> Dict[str, Union[int, float]]:
        loader = ExperimentLoader(path)
        data = loader.get_experiment_final_df()
        analyser = TextAnalyser(data)
        data_metrics = analyser.analyse_data().to_dict()["Value"]
        data_metrics.update(loader.extract_metrics())
        return data_metrics

    def load_experiment_metrics(self):
        """Loads and analyses metrics for all experiments."""
        for path in self.experiment_paths:
            loader = ExperimentLoader(path)
            identifier = (
                loader.experiment_path.name
            )  # Assuming the folder name is the identifier
            self._experiment_metrics[identifier] = self._load_and_analyse_experiment(
                path
            )

    def load_real_dataset_metrics(self):
        """Loads and analyses metrics for the real dataset."""
        if self.real_dataset is None:
            raise ValueError("Real dataset not provided.")
        analyser = TextAnalyser(self.real_dataset)
        self._real_dataset_metrics = analyser.analyse_data().to_dict()["Value"]

    def compare_metrics(self) -> pd.DataFrame:
        if not self._experiment_metrics or not self._real_dataset_metrics:
            raise ValueError("Metrics not loaded. Use load methods first.")

        avg_real_metrics = pd.Series(self._real_dataset_metrics)
        differences = {}
        for identifier, metrics in self._experiment_metrics.items():
            differences[identifier] = pd.Series(metrics).sub(avg_real_metrics)

        return pd.DataFrame(differences)

    def aggregate_metrics(self) -> pd.DataFrame:
        experiments_df = self._formatted_experiment_metrics_df()
        real_df = self._formatted_real_dataset_metrics_df()

        # Ensure that both DataFrames have the exact same columns.
        aggregated_df = self._consistent_column_concat([experiments_df, real_df])

        return aggregated_df

    def _formatted_experiment_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._experiment_metrics).T

    def _formatted_real_dataset_metrics_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._real_dataset_metrics, index=["Real"])

    def _consistent_column_concat(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate DataFrames ensuring consistent columns."""
        all_cols = sorted(set().union(*[df.columns for df in dfs]))

        for df in dfs:
            for col in all_cols:
                if col not in df:
                    df[col] = np.nan

        return pd.concat(dfs, axis=0)

    def visualise_metrics(self):
        # This can be filled with specific visualization code (e.g., using matplotlib or seaborn)
        pass
