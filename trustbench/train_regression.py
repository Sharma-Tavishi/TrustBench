from sklearn.isotonic import IsotonicRegression
import pandas as pd
import numpy as np
import argparse

all_metrics = {'metrics': ['rougue-l', 'f1'],
               'nli': ['nli_entailment', 'nli_contradiction', 'nli_neutral'],
               'consistency': ['ng1_prec','ng1_rec','ng1_f1']}


def parser():
    parser = argparse.ArgumentParser(
        description="Train a Isotonic Regression over model metrics.",
        formatter_class=argparse.RawTextHelpFormatter # For better help text formatting
    )

    # 1. Positional argument for the model name
    parser.add_argument(
        "--model",'-m',
        type=str,
        required=True,
        help="The name of the model to train (e.g., 'llama2', 'gpt')."
    )

    parser.add_argument(
        "--dataset",'-d',
        type=str,
        required=True,
        choices=['truthful_qa', 'mixed_qa', 'med_qa', 'fin_qa'],
        help="Name of dataset from ('truthful_qa', 'mixed_qa', 'med_qa', 'fin_qa')."
    )

    # 2. Positional argument for the metric with a specific list of choices
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Train regression over all metrics. (default: False)"
    )

    parser.add_argument(
        "--metric",
        type=str,
        choices=['rougue-l', 'f1', 'nli_entailment', 'nli_contradiction', 'nli_neutral','ng1_prec','ng1_rec','ng1_f1'],
        help="The specific evaluation metric to use. Required if --all-metrics is not specified."
    )

    # 4. Optional argument for an output directory
    parser.add_argument(
        "--model-save",
        type=str,
        default="./saved_models/",
        help="The directory to save trained model. (default: ./saved_models/)"
    )
    # 5. Optional argument for isotonic regression y_min
    parser.add_argument(
        '--y_min',
        type=float,
        default=None,
        help='The minimum value of the isotonic regression function. (optional)'
    )    
    # 6. Optional argument for isotonic regression y_max
    parser.add_argument(
        '--y_max',
        type=float,
        default=None,
        help='The maximum value of the isotonic regression function. (optional)'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    print(args)