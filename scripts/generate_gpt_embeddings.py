import argparse
from pathlib import Path

import pandas as pd

from instasynth import embedding_generation
from instasynth.config import logger


def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(description="Generate GPT embeddings for captions")
    parser.add_argument(
        "--storage_path",
        default="embeddings/",
        help="Path to store the embeddings",
    )
    parser.add_argument(
        "--embedding_file_name",
        default="embeddings.pkl",
        help="File name to store the embeddings",
    )
    parser.add_argument(
        "--dataframe_path",
        help="Path to the dataframe containing the captions",
    )
    args = parser.parse_args()
    # Create storage path if necessary
    storage_path = Path("embeddings/")
    storage_path.mkdir(parents=True, exist_ok=True)
    # Load dataframe
    df = pd.read_pickle(args.dataframe_path)
    # Generate embeddings
    storage = embedding_generation.EmbeddingStorage(
        storage_path=storage_path, embedding_file_name=args.embedding_file_name
    )
    gen = embedding_generation.EmbeddingGenerator(
        storage, df["caption"].str.lower().tolist()
    )
    logger.info(f"Generating embeddings for {len(df)} captions!")
    gen.generate_and_store()


if __name__ == "__main__":
    main()
