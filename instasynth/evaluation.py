import re
import json
import string
from typing import Dict, List, Tuple, ClassVar, Set, Union, Optional
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import textstat
import emoji
import pandas as pd
import numpy as np
import faiss
import networkx as nx
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import RFE
from nltk import ngrams
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from rouge import Rouge


from .embedding_generation import EmbeddingStorage, EmbeddingGenerator
from .utils import format_text


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
            "pct_unique_captions": len(self.data["caption"].str.lower().unique())
            / len(self.data),
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

    def _compute_jaccard_similarity(self, other: "TextAnalyser", n: int) -> float:
        # Get n-grams for both datasets
        self_ngrams = set(self._extract_ngrams(n))
        other_ngrams = set(other._extract_ngrams(n))
        # Compute Jaccard similarity
        intersection = len(self_ngrams.intersection(other_ngrams))
        union = len(self_ngrams.union(other_ngrams))

        return intersection / union if union != 0 else 0

    def _tag_overlap_metrics(self, other: "TextAnalyser") -> Dict[str, float]:
        metrics = {}
        tag_identifiers = {"#": "hashtag", "@": "user_tag"}
        self_vocab = {
            tag: set([w for w in self.vocabulary if w.startswith(tag)])
            for tag in tag_identifiers
        }
        other_vocab = {
            tag: set([w for w in other.vocabulary if w.startswith(tag)])
            for tag in tag_identifiers
        }
        for tag, identifier in tag_identifiers.items():
            intersection = len(self_vocab[tag].intersection(other_vocab[tag]))
            union = len(self_vocab[tag].union(other_vocab[tag]))
            metrics[f"{identifier}_overlap"] = intersection / union if union != 0 else 0
        return metrics

    def compare_jaccard_similarity(self, other: "TextAnalyser", n: int) -> float:
        """Compare the current dataset with another using Jaccard similarity."""
        return self._compute_jaccard_similarity(other, n)

    def _pronoun_metrics(self) -> Dict[str, float]:
        metrics = {}
        for k in ["first", "second", "third"]:
            metrics[f"pct_{k}_person_pronouns"] = self.pronoun_person_frequency[
                k
            ] / sum(self.pronoun_person_frequency.values())
        return metrics

    def analyse_data(self, compare_ta: "TextAnalyser" = None) -> pd.DataFrame:
        metrics = self._basic_metrics()
        metrics.update(self._pattern_based_metrics())
        metrics.update(self._text_complexity_metrics())
        metrics.update(self._ngram_metrics())
        metrics.update(self._pronoun_metrics())
        if compare_ta is not None:
            for n in range(1, 4):
                metrics[
                    f"jaccard_similarity_{n}gram"
                ] = self.compare_jaccard_similarity(compare_ta, n)
            metrics.update(self._tag_overlap_metrics(compare_ta))
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
        return metrics_df


@dataclass
class TopicModellingAnalyser:
    data: pd.DataFrame
    num_topics: int = 10
    _corpus: list = field(default_factory=list, repr=False)
    _id2word: corpora.Dictionary = field(default=None, repr=False)
    _lda_model: models.LdaModel = field(default=None, repr=False)
    _tokenizer: TweetTokenizer = field(default_factory=TweetTokenizer, repr=False)
    __stopwords: ClassVar[Set[str]] = set(stopwords.words("english"))

    def __post_init__(self):
        if "caption" not in self.data.columns:
            raise ValueError("DataFrame must contain 'caption' column.")
        if "type" not in self.data.columns:
            raise ValueError(
                "DataFrame must contain 'type' column indicating 'real' or 'synthetic'."
            )
        self.prepare_data()

    def prepare_data(self) -> None:
        self.data["tokenized"] = (
            self.data["caption"]
            .str.lower()
            .apply(
                lambda x: [
                    w
                    for w in self._tokenizer.tokenize(x)
                    if w not in self.__stopwords and w not in string.punctuation
                ]
            )
        )
        self._id2word = corpora.Dictionary(self.data["tokenized"])
        self._corpus = [self._id2word.doc2bow(text) for text in self.data["tokenized"]]

    def build_lda_model(self) -> None:
        self._lda_model = models.LdaModel(
            corpus=self._corpus, id2word=self._id2word, num_topics=self.num_topics
        )

    def get_topics(self) -> List[Tuple[int, str]]:
        if not self._lda_model:
            raise ValueError(
                "LDA Model has not been built. Run build_lda_model() first."
            )
        return self._lda_model.print_topics()

    def coherence_score(self) -> float:
        coherence_model = CoherenceModel(
            model=self._lda_model,
            texts=self.data["tokenized"],
            dictionary=self._id2word,
            coherence="c_v",
        )
        return coherence_model.get_coherence()

    def compare_topics(self, real_or_synthetic: str) -> List[Tuple[int, str]]:
        filtered_data = self.data[self.data["type"] == real_or_synthetic]
        id2word = corpora.Dictionary(filtered_data["tokenized"])
        corpus = [id2word.doc2bow(text) for text in filtered_data["tokenized"]]
        lda_model = models.LdaModel(
            corpus=corpus, id2word=id2word, num_topics=self.num_topics
        )
        return lda_model.print_topics()

    def analyse_data(self) -> Dict[str, Union[float, List[Tuple[int, str]]]]:
        self.build_lda_model()
        metrics = {
            "All Topics": self.get_topics(),
            "Real Data Topics": self.compare_topics("real"),
            "Synthetic Data Topics": self.compare_topics("synthetic"),
            "Coherence Score": self.coherence_score(),
        }
        return metrics


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
class NetworkAnalyser:
    data: pd.DataFrame
    _hashtag_network: nx.Graph = field(default=None, repr=False, init=False)
    _usertag_network: nx.Graph = field(default=None, repr=False, init=False)
    _hashtag_usertag_network: nx.Graph = field(default=None, repr=False, init=False)
    __HASHTAG_PATTERN: ClassVar[re.Pattern] = re.compile(r"#(\w+)")
    __USER_TAG_PATTERN: ClassVar[re.Pattern] = re.compile(r"@(\w+)")

    def __post_init__(self):
        if "caption" not in self.data.columns:
            raise ValueError("DataFrame must contain 'caption' column.")
        # Creating the corresponding tag columns
        self.data["hashtags"] = self.data["caption"].apply(
            self._extract_from_pattern, pattern=self.__HASHTAG_PATTERN
        )
        self.data["usertags"] = self.data["caption"].apply(
            self._extract_from_pattern, pattern=self.__USER_TAG_PATTERN
        )

    @staticmethod
    def _extract_from_pattern(caption: str, pattern: re.Pattern) -> list:
        return pattern.findall(caption)

    @staticmethod
    def _create_cooccurrence_network(tags_list: pd.Series) -> nx.Graph:
        G = nx.Graph()
        for tags in tags_list:
            for i, tag in enumerate(tags):
                if G.has_node(tag):
                    G.nodes[tag]["count"] = G.nodes[tag].get("count", 0) + 1
                else:
                    G.add_node(tag, count=1)
                for j in range(i + 1, len(tags)):
                    if G.has_edge(tag, tags[j]):
                        G[tag][tags[j]]["weight"] += 1
                    else:
                        G.add_edge(tag, tags[j], weight=1)
        return G

    @staticmethod
    def _create_cooccurrence_bipartite_network(
        hashtags: pd.Series, usertags: pd.Series
    ) -> nx.Graph:
        G = nx.Graph()
        for i in range(len(hashtags)):
            hashtag_list = hashtags.iloc[i]
            usertag_list = usertags.iloc[i]

            # Add hashtag nodes to the network with a type attribute
            for hashtag in hashtag_list:
                if not G.has_node(hashtag):
                    G.add_node(hashtag, type="hashtag", count=1)
                else:
                    G.nodes[hashtag]["count"] = G.nodes[hashtag].get("count", 0) + 1

            # Add usertag nodes to the network with a type attribute
            for usertag in usertag_list:
                if not G.has_node(usertag):
                    G.add_node(usertag, type="usertag")

            # Create edges between cooccurring hashtags and usertags
            for hashtag in hashtag_list:
                for usertag in usertag_list:
                    if G.has_edge(hashtag, usertag):
                        G[hashtag][usertag]["weight"] += 1
                    else:
                        G.add_edge(hashtag, usertag, weight=1)

        return G

    @property
    def hashtag_network(self):
        if self._hashtag_network is None:
            self._hashtag_network = self._create_cooccurrence_network(
                self.data["hashtags"]
            )
        return self._hashtag_network

    @property
    def usertag_network(self):
        if self._usertag_network is None:
            self._usertag_network = self._create_cooccurrence_network(
                self.data["usertags"]
            )
        return self._usertag_network

    @property
    def hashtag_usertag_network(self):
        if self._hashtag_usertag_network is None:
            self._hashtag_usertag_network = self._create_cooccurrence_bipartite_network(
                self.data["hashtags"], self.data["usertags"]
            )
        return self._hashtag_usertag_network

    def _get_network_metrics(self, G: nx.Graph) -> dict:
        metrics = {
            "number_of_nodes": G.number_of_nodes(),
            "number_of_edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_clustering_coefficient": nx.average_clustering(G),
            "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "avg_betweenness_centrality": sum(nx.betweenness_centrality(G).values())
            / G.number_of_nodes(),
            "avg_closeness_centrality": sum(nx.closeness_centrality(G).values())
            / G.number_of_nodes(),
            "avg_eigenvector_centrality": sum(
                nx.eigenvector_centrality_numpy(G).values()
            )
            / G.number_of_nodes(),
            "assortativity": nx.degree_assortativity_coefficient(G),
            "transitivity": nx.transitivity(G),
        }

        return metrics

    def analyse_data(self) -> pd.DataFrame:
        metrics = {}
        networks = {
            "hashtag": self.hashtag_network,
            "usertag": self.usertag_network,
            "hashtag_usertag": self.hashtag_usertag_network,
        }
        for name, G in networks.items():
            if not nx.is_empty(G):
                network_metrics = self._get_network_metrics(G)
                for metric, value in network_metrics.items():
                    metrics[f"{name}_{metric}"] = value
        metrics_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
        return metrics_df


@dataclass
class NgramSimilarityAnalyser:
    real_posts: List[str] = field(repr=False)
    synthetic_posts: List[str] = field(repr=False)
    _rouge: Rouge = field(default_factory=Rouge, init=False, repr=False)

    def compute_rouge_n(self, n: int) -> List[float]:
        scores = []
        for real, synthetic in zip(self.real_posts, self.synthetic_posts):
            score = self._rouge.get_scores(synthetic, real, avg=True)[f"rouge-{n}"]["f"]
            scores.append(score)
        return scores

    def _compute_internal_rouge_n(self, posts: List[str], n: int) -> float:
        total_scores = 0
        total_pairs = 0

        for i, post1 in enumerate(posts):
            for j, post2 in enumerate(posts):
                if i != j:  # Avoid comparing the post with itself
                    score = self._rouge.get_scores(post1, post2, avg=True)[
                        f"rouge-{n}"
                    ]["f"]
                    total_scores += score
                    total_pairs += 1

        return total_scores / total_pairs if total_pairs != 0 else 0

    def _internal_ngram_similarity_metrics(self, max_n: int) -> Dict[str, float]:
        metrics = {}

        for n in range(1, max_n + 1):
            real_internal_score = self._compute_internal_rouge_n(self.real_posts, n)
            synthetic_internal_score = self._compute_internal_rouge_n(
                self.synthetic_posts, n
            )

            metrics[f"avg_internal_rouge_{n}_real"] = real_internal_score
            metrics[f"avg_internal_rouge_{n}_synthetic"] = synthetic_internal_score

        return metrics

    def _ngram_similarity_metrics(self, n: int) -> Dict[str, float]:
        scores = self.compute_rouge_n(n)

        metrics = {
            f"avg_rouge_{n}": np.mean(scores),
            f"median_rouge_{n}": np.median(scores),
            f"std_rouge_{n}": np.std(scores),
            f"q1_rouge_{n}": np.percentile(scores, 25),
            f"q3_rouge_{n}": np.percentile(scores, 75),
            f"min_rouge_{n}": np.min(scores),
            f"max_rouge_{n}": np.max(scores),
        }

        return metrics

    def analyse_ngram_similarity(
        self, max_n: int = 2, analyse_internal: bool = True
    ) -> Dict[str, float]:
        metrics = {}
        for n in range(1, max_n + 1):
            metrics.update(self._ngram_similarity_metrics(n))
        if analyse_internal:
            metrics.update(self._internal_ngram_similarity_metrics(max_n))
        return metrics


@dataclass
class EmbeddingSimilarityAnalyser:
    embeddings_storage: EmbeddingStorage
    real_posts: List[str] = field(repr=False)
    synthetic_posts: List[str] = field(repr=False)
    _real_emb_matrix: np.array = field(default=None, init=False, repr=False)
    _synthetic_emb_matrix: np.array = field(default=None, init=False, repr=False)
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
    def synthetic_emb_matrix(self) -> np.array:
        if self._synthetic_emb_matrix is None:
            synthetic_emb_data = [
                self.embeddings_storage.get_embedding(post)
                for post in self.synthetic_posts
            ]
            self._synthetic_emb_matrix = np.array(synthetic_emb_data).astype("float32")
        return self._synthetic_emb_matrix

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

    def _compute_internal_similarity(self, emb_matrix: np.array) -> float:
        normalized_embeddings = self._normalize(np.copy(emb_matrix))
        similarity_matrix = normalized_embeddings @ normalized_embeddings.T
        np.fill_diagonal(similarity_matrix, np.nan)
        return np.nanmean(similarity_matrix)

    def _internal_similarity_metrics(self) -> Dict[str, float]:
        return {
            "real_internal_cosine_sim": self._compute_internal_similarity(
                self.real_emb_matrix
            ),
            "synthetic_internal_cosine_sim": self._compute_internal_similarity(
                self.synthetic_emb_matrix
            ),
        }

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

    def analyse_similarity(
        self,
        analyse_top_k_recall: bool = True,
        analyse_internal_similarity: bool = True,
    ) -> Dict[str, float]:
        metrics = self._cosine_similarity_metrics()
        if analyse_top_k_recall:
            for k in [1, 10, 100]:
                metrics[f"top_{k}_recall_cosine_sim"] = self._top_k_recall(k)
        if analyse_internal_similarity:
            metrics.update(self._internal_similarity_metrics())
        return metrics


@dataclass
class SingleExperimentAnalyser:
    data: pd.DataFrame = field(repr=False)

    def __post_init__(self):
        # Fixing hashtags with multiple consecutive #, usually generated because of the #### separator
        self.data["caption"] = self.data["caption"].str.replace(r"#{2,}", "#")
        self.data["caption"] = self.data["caption"].str.lower()

    def analyse_experiment(
        self,
        real_dataset: pd.DataFrame = None,
        metrics: dict = None,
        test_dataset_ads: pd.DataFrame = None,
        test_dataset_ads_undisclosed: pd.DataFrame = None,
        embedding_storage: EmbeddingStorage = None,
        analyse_embeddings: bool = True,
        analyse_top_k_recall: bool = True,
        analyse_internal_similarity: bool = True,
    ) -> Dict[str, float]:
        analyser = TextAnalyser(self.data)
        real_data_ta = TextAnalyser(real_dataset) if real_dataset is not None else None
        # Basic metrics
        data_metrics = analyser.analyse_data(compare_ta=real_data_ta).to_dict()["Value"]
        # Classification metrics
        if test_dataset_ads is not None:
            classifier = ClassificationAnalyser(
                data=self.data,
                evaluation_data=test_dataset_ads,
                evaluation_data_ann=test_dataset_ads_undisclosed,
            )
            data_metrics.update(classifier.ad_detection_performance())
        # Network metrics
        nw_analyser = NetworkAnalyser(self.data)
        nw_metrics = nw_analyser.analyse_data().to_dict()["Value"]
        data_metrics.update({f"NW_{k}": v for k, v in nw_metrics.items()})
        # Embedding similarity metrics
        if embedding_storage is not None and analyse_embeddings:
            data_metrics.update(
                self._analyse_embedding_metrics(
                    real_dataset=real_dataset,
                    embedding_storage=embedding_storage,
                    analyse_top_k_recall=analyse_top_k_recall,
                    analyse_internal_similarity=analyse_internal_similarity,
                )
            )
        if metrics:
            data_metrics.update(metrics)

        return data_metrics

    def _analyse_embedding_metrics(
        self,
        real_dataset: pd.DataFrame,
        embedding_storage: EmbeddingStorage,
        analyse_top_k_recall: bool = True,
        analyse_internal_similarity: bool = True,
    ) -> Dict[str, float]:
        real_posts = real_dataset["caption"].apply(format_text).tolist()
        synthetic_posts = self.data["caption"].apply(format_text).tolist()
        EmbeddingGenerator(
            embedding_storage, texts=real_posts + synthetic_posts, verbose=False
        ).generate_and_store()
        embedding_analyser = EmbeddingSimilarityAnalyser(
            embeddings_storage=embedding_storage,
            real_posts=real_posts,
            synthetic_posts=synthetic_posts,
        )
        return embedding_analyser.analyse_similarity(
            analyse_top_k_recall=analyse_top_k_recall,
            analyse_internal_similarity=analyse_internal_similarity,
        )


@dataclass
class ExperimentEvaluator:
    experiment_paths: List[Path]
    # Real dataset to compare with
    real_dataset: Optional[pd.DataFrame] = None
    # Test dataset to evaluate ad detection performance
    test_dataset_ads: Optional[pd.DataFrame] = None
    # Test dataset to evaluate ad detection performance on undisclosed ads
    test_dataset_ads_undisclosed: Optional[pd.DataFrame] = None
    embedding_storage: Optional[EmbeddingStorage] = None
    _embedding_generator: Optional[EmbeddingGenerator] = field(default=None, init=False)
    _experiment_metrics: Dict[str, Dict[str, Union[int, float]]] = field(
        default_factory=dict, init=False
    )
    _real_dataset_metrics: Dict[str, Union[int, float]] = field(
        default_factory=dict, init=False
    )

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

    def _load_and_analyse_experiment(self, path: Path) -> Dict[str, Union[int, float]]:
        loader = ExperimentLoader(path)
        data = loader.get_experiment_final_df().dropna().query("caption != ''")
        data_metrics = SingleExperimentAnalyser(data=data).analyse_experiment(
            real_dataset=self.real_dataset,
            test_dataset_ads=self.test_dataset_ads,
            test_dataset_ads_undisclosed=self.test_dataset_ads_undisclosed,
            embedding_storage=self.embedding_storage,
        )
        data_metrics.update(loader.extract_metrics())
        return data_metrics

    def load_real_dataset_metrics(self):
        if self.real_dataset is None:
            raise ValueError("Real dataset not provided.")
        self._real_dataset_metrics = SingleExperimentAnalyser(
            data=self.real_dataset
        ).analyse_experiment(
            test_dataset_ads=self.test_dataset_ads,
            test_dataset_ads_undisclosed=self.test_dataset_ads_undisclosed,
        )

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
