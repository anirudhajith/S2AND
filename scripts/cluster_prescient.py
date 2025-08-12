"""
Try running clustering with the trained model on PreScient data.
"""

import os
import sys
import argparse

import pickle5 as pickle; import sys; sys.setrecursionlimit(2000000000)
from s2and.data import ANDData
from s2and.eval import cluster_eval
from s2and.consts import FEATURIZER_VERSION, DEFAULT_CHUNK_SIZE, PROJECT_ROOT_PATH
from s2and.featurizer import FeaturizationInfo, featurize
import numpy as np
from s2and.model import PairwiseModeler, Clusterer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster PreScient data with trained S2AND model.")
    parser.add_argument("--papers_path", type=str, required=True, help="Path to the PreScient papers JSON file.")
    parser.add_argument("--signatures_path", type=str, required=True, help="Path to the PreScient signatures JSON file.")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Path to the SPECTER embeddings pickle file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model pickle file.")
    parser.add_argument("--dataset_name", type=str, default="prescient", help="Name of the dataset being processed.")
    parser.add_argument("--n_jobs", type=int, default=32, help="Number of parallel jobs to run.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output clusters.")
    args = parser.parse_args()

    assert args.output_path.endswith(".pickle"), "Output path must end with .pickle"
    os.environ["OMP_NUM_THREADS"] = str(args.n_jobs)

    # to train the pairwise model, we define which feature categories to use
    # here it is all of them
    features_to_use = [
        "name_similarity",
        "affiliation_similarity",
        "email_similarity",
        "coauthor_similarity",
        "venue_similarity",
        "year_diff",
        "title_similarity",
        "reference_features",
        "misc_features",
        "name_counts",
        "embedding_similarity",
        "journal_similarity",
        "advanced_name_similarity",
    ]

    # we also have this special second "nameless" model that doesn't use any name-based features
    # it helps to improve clustering performance by preventing model overreliance on names
    nameless_features_to_use = [
        feature_name
        for feature_name in features_to_use
        if feature_name not in {"name_similarity", "advanced_name_similarity", "name_counts"}
    ]

    # we store all the information about the features in this convenient wrapper
    featurization_info = FeaturizationInfo(features_to_use=features_to_use, featurizer_version=FEATURIZER_VERSION)
    nameless_featurization_info = FeaturizationInfo(
        features_to_use=nameless_features_to_use, featurizer_version=FEATURIZER_VERSION
    )

    # this is the prod 1.1 model, which we may or may not retrain
    with open(args.model_path, "rb") as f:
        clusterer = pickle.load(f)["clusterer"]
        clusterer.use_cache = False  # very important for this experiment!!!

    anddata = ANDData(
        signatures=args.signatures_path,
        papers=args.papers_path,
        name=args.dataset_name,
        mode="inference",
        specter_embeddings=args.embeddings_path,
        clusters=None,
        block_type="s2",
        train_pairs=None,
        val_pairs=None,
        test_pairs=None,
        train_pairs_size=None,
        val_pairs_size=None,
        test_pairs_size=None,
        n_jobs=args.n_jobs,
        load_name_counts=True,
        preprocess=True,
        random_seed=args.random_seed,
        name_tuples=None,
    )

    block_dict = anddata.get_blocks()
    pred_clusters, _ = clusterer.predict(block_dict, anddata, use_s2_clusters=False)

    print(f"Writing clusters to {args.output_path}")
    with open(args.output_path, "wb") as f:
        pickle.dump(pred_clusters, f, protocol=pickle.HIGHEST_PROTOCOL)