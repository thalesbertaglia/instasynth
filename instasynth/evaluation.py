import re
import json
from typing import Dict, List, Tuple, ClassVar, Set, Union, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import textstat
import emoji
import pandas as pd
import numpy as np
import faiss
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


from .embedding_generation import EmbeddingStorage


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
    remove_stopwords: bool = False
    _tokenized_captions: pd.Series = None
    _ngrams: Dict[int, List[str]] = field(default_factory=dict, repr=False)
    _vocabulary: Set[str] = field(default=None, repr=False)
    _pronoun_frequency: Dict[str, Counter] = field(default=None, repr=False)
    _stopwords: Set[str] = field(default_factory=set, repr=False)
    __HASHTAG_PATTERN: ClassVar[re.Pattern] = re.compile(r"#(\w+)")
    __USER_TAG_PATTERN: ClassVar[re.Pattern] = re.compile(r"@(\w+)")
    __PRONOUNS: ClassVar[Dict] = {
        "first_singular": {"i", "me", "my", "mine"},
        "first_plural": {"we", "us", "our", "ours"},
        "second_singular": {"you", "your", "yours"},
        "second_plural": {"you", "your", "yours"},
        "third_singular": {"he", "him", "his", "she", "her", "hers", "it", "its"},
        "third_plural": {"they", "them", "their", "theirs"},
    }

    def __post_init__(self):
        if "caption" not in self.data.columns:
            raise ValueError("DataFrame must contain 'caption' column.")

        self._stopwords = set(stopwords.words("english"))
        self.tokenizer = TweetTokenizer(preserve_case=False)
        # Fixing hashtags with multiple consecutive #, usually generated because of the #### separator
        self.data["caption"] = self.data["caption"].str.replace(r"#{2,}", "#")

    @property
    def tokenized_captions(self) -> pd.Series:
        if self._tokenized_captions is None:
            tokens = self.data["caption"].apply(self.tokenizer.tokenize)
            if self.remove_stopwords:
                tokens = tokens.apply(
                    lambda x: [w for w in x if w not in self._stopwords]
                )
            self._tokenized_captions = tokens
        return self._tokenized_captions

    @property
    def vocabulary(self) -> Set[str]:
        if self._vocabulary is None:
            self._vocabulary = set(self.tokenized_captions.sum())
        return self._vocabulary

    @property
    def pronoun_frequency(self) -> Dict[str, Counter]:
        if self._pronoun_frequency is None:
            self._pronoun_frequency = self._compute_pronoun_frequency()
        return self._pronoun_frequency

    @property
    def pronoun_category_frequency(self) -> Dict[str, int]:
        return {
            category: sum(pronouns_freq.values())
            for category, pronouns_freq in self.pronoun_frequency.items()
        }

    @property
    def pronoun_person_frequency(self) -> Dict[str, int]:
        return {
            person: sum(
                val
                for cat, val in self.pronoun_category_frequency.items()
                if cat.startswith(person)
            )
            for person in ["first", "second", "third"]
        }

    def _compute_pronoun_frequency(self) -> Dict[str, Counter]:
        pronoun_count = {category: Counter() for category in self.__PRONOUNS}
        unigram_counts = dict(self._get_top_ngrams(n=1, top_k=len(self.vocabulary)))
        reverse_pronoun_lookup = {
            pronoun: category
            for category, pronouns in self.__PRONOUNS.items()
            for pronoun in pronouns
        }
        for unigram, count in unigram_counts.items():
            unigram = unigram[0]
            if unigram in reverse_pronoun_lookup:
                pronoun_count[reverse_pronoun_lookup[unigram]][unigram] += count

        return pronoun_count

    def _extract_ngrams(self, n: int) -> List[str]:
        if n not in self._ngrams:
            self._ngrams[n] = [
                ngram
                for caption in self.tokenized_captions
                for ngram in ngrams(caption, n)
            ]
        return self._ngrams[n]

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

    def _basic_metrics(self) -> Dict[str, float]:
        return {
            "avg_caption_length": self.tokenized_captions.apply(len).mean(),
            "std_caption_length": self.tokenized_captions.apply(len).std(),
            "vocabulary_size": len(self.vocabulary),
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

    def _ngram_metrics(self) -> Dict[str, float]:
        metrics = {}
        for n in [1, 2, 3]:
            ngrams = self._extract_ngrams(n)
            metrics[f"avg_{n}gram_per_post"] = len(ngrams) / len(self.data)
            metrics[f"n_unique_{n}gram"] = len(set(ngrams))
        return metrics

    def _get_top_ngrams(self, n: int, top_k: int = 1000) -> List[Tuple[str, int]]:
        return Counter(self._extract_ngrams(n)).most_common(top_k)

    def _pronoun_metrics(self) -> Dict[str, float]:
        metrics = {}
        for k in ["first", "second", "third"]:
            metrics[f"pct_{k}_person_pronouns"] = self.pronoun_person_frequency[
                k
            ] / sum(self.pronoun_person_frequency.values())
        return metrics

    def analyse_data(self) -> pd.DataFrame:
        metrics = self._basic_metrics()
        metrics.update(self._pattern_based_metrics())
        metrics.update(self._text_complexity_metrics())
        metrics.update(self._ngram_metrics())
        metrics.update(self._pronoun_metrics())
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
        return metrics_df


@dataclass
class ClassificationAnalyser:
    data: pd.DataFrame
    evaluation_data: pd.DataFrame
    # Annotated data to evaluate ad detection performance on undisclosed posts
    evaluation_data_ann: Optional[pd.DataFrame] = None
    text_column: str = "caption"
    target_column: str = "sponsorship"

    __VECTORIZER: ClassVar[TfidfVectorizer] = TfidfVectorizer()
    __MODEL: ClassVar[LogisticRegression] = LogisticRegression(random_state=42)

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

    def evaluate(self, X_test, y_test, identifier: str) -> Dict[str, float]:
        predictions = self.__MODEL.predict(X_test)
        metrics = {
            f"{identifier}_accuracy": accuracy_score(y_test, predictions),
            f"{identifier}_precision": precision_score(y_test, predictions),
            f"{identifier}_recall": recall_score(y_test, predictions),
            f"{identifier}_f1": f1_score(y_test, predictions),
        }
        return metrics

    def ad_detection_performance(self) -> Dict[str, float]:
        X_train, y_train = self._preprocess(self.data)
        X_test, y_test = self._preprocess(self.evaluation_data)
        self.train(X_train, y_train)
        performance_metrics = self.evaluate(X_test, y_test, "ad_detection")
        if self.evaluation_data_ann is not None:
            X_test_ann, y_test_ann = self._preprocess(self.evaluation_data_ann)
            performance_metrics["ad_detection_undisclosed_accuracy"] = self.evaluate(
                X_test_ann, y_test_ann, "ad_detection_undisclosed"
            )["ad_detection_undisclosed_accuracy"]
        return performance_metrics


@dataclass
class EmbeddingSimilarityAnalyser:
    embeddings_storage: EmbeddingStorage
    real_posts: List[str] = field(repr=False)
    synthetic_posts: List[str] = field(repr=False)
    _real_emb_matrix: np.array = field(default=None, init=False, repr=False)
    _index: faiss.Index = field(default=None, init=False, repr=False)

    @property
    def real_emb_matrix(self) -> np.array:
        if self._real_emb_matrix is None:
            real_emb_data = [
                self.embeddings_storage.get_embedding(post) for post in self.real_posts
            ]
            self._real_emb_matrix = np.array(real_emb_data).astype("float32")
        return self._real_emb_matrix

    @property
    def index(self) -> faiss.Index:
        if self._index is None:
            d = self.real_emb_matrix.shape[1]
            self._index = faiss.IndexFlatIP(d)
            normalized_embeddings = self._normalize(
                np.copy(self.real_emb_matrix)
            )  # Copying before normalisation
            self._index.add(normalized_embeddings.astype("float32"))
        return self._index

    def _normalize(self, embeddings: np.array) -> np.array:
        """Normalize the embeddings to make them unit vectors."""
        faiss.normalize_L2(embeddings)
        return embeddings

    def compute_similarity(self, k=1) -> Tuple[np.array, np.array]:
        synthetic_emb_data = [
            self.embeddings_storage.get_embedding(post) for post in self.synthetic_posts
        ]
        synthetic_embeddings = self._normalize(
            np.array(synthetic_emb_data).astype("float32")
        )
        # Search normalized embeddings
        distances, indices = self.index.search(synthetic_embeddings, k)
        return distances, indices

    def get_top_n_similar(self, n=5) -> np.array:
        _, indices = self.compute_similarity(n)
        return indices

    def _top_k_recall(self, k=5) -> float:
        """
        Computes the Top-k Recall. It checks if the corresponding real caption is in the top-k
        most similar real captions for a synthetic one and then computes the fraction for which this is true.
        """
        indices = self.get_top_n_similar(k)
        recall_hits = sum([i in idx for i, idx in enumerate(indices)])
        return recall_hits / len(self.synthetic_posts)

    def _cosine_similarity_metrics(self) -> Dict[str, float]:
        """
        Computes statistics of the cosine similarity distribution.
        """
        distances, _ = self.compute_similarity(1)

        metrics = {
            "avg_cosine_sim": np.mean(distances),
            "median_cosine_sim": np.median(distances),
            "std_cosine_sim": np.std(distances),
            "q1_cosine_sim": np.percentile(distances, 25),
            "q3_cosine_sim": np.percentile(distances, 75),
            "min_cosine_sim": np.min(distances),
            "max_cosine_sim": np.max(distances),
        }

        return metrics

    def analyse_similarity(self) -> Dict[str, float]:
        metrics = self._cosine_similarity_metrics()
        for k in [1, 10, 100]:
            metrics[f"top_{k}_recall_cosine_sim"] = self._top_k_recall(k)
        return metrics


@dataclass
class ExperimentEvaluator:
    experiment_paths: List[Path]
    # Real dataset to compare with
    real_dataset: Optional[pd.DataFrame] = None
    # Test dataset to evaluate ad detection performance
    test_dataset_ads: Optional[pd.DataFrame] = None
    # Test dataset to evaluate ad detection performance on undisclosed ads
    test_dataset_ads_undisclosed: Optional[pd.DataFrame] = None

    _experiment_metrics: Dict[str, Dict[str, Union[int, float]]] = field(
        default_factory=dict, init=False
    )
    _real_dataset_metrics: Dict[str, Union[int, float]] = field(
        default_factory=dict, init=False
    )

    def _load_and_analyse_experiment(self, path: Path) -> Dict[str, Union[int, float]]:
        loader = ExperimentLoader(path)
        data = loader.get_experiment_final_df().dropna()
        analyser = TextAnalyser(data)
        data_metrics = analyser.analyse_data().to_dict()["Value"]
        if self.test_dataset_ads is not None:
            classifier = ClassificationAnalyser(
                data=data,
                evaluation_data=self.test_dataset_ads,
                evaluation_data_ann=self.test_dataset_ads_undisclosed,
            )
            data_metrics.update(classifier.ad_detection_performance())
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
        if self.test_dataset_ads is not None:
            classifier = ClassificationAnalyser(
                data=self.real_dataset,
                evaluation_data=self.test_dataset_ads,
                evaluation_data_ann=self.test_dataset_ads_undisclosed,
            )
            self._real_dataset_metrics.update(classifier.ad_detection_performance())

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
