from sklearn.isotonic import IsotonicRegression
import pandas as pd
import numpy as np
import argparse

import os
import pickle
import json
import tqdm


all_metric_list = ['truthfulness_score' ,'informative' ,'consistency']

class RegressionTrainer:
    def __init__(self, eval_results:pd.DataFrame, 
                y_min=None, y_max=None):
        self.model = IsotonicRegression(y_min=y_min, y_max=y_max)

        self.eval_results = eval_results
        # Convert score to numeric, coerce errors to NaN, then fill NaN with 1
        self.eval_results['score'] = pd.to_numeric(self.eval_results['score'], errors='coerce').fillna(1)
        ## Set dummy values for X and y
        self.X = self.eval_results[['score']].values
        self.y = None
        self.y_pred = None

    def set_metric(self, metric_label:str):
        self.y = pd.to_numeric(self.eval_results[metric_label].ffill()).values

    def train(self):
        if self.y is None:
            raise ValueError("Training y must be set before training.")
        self.y_pred = self.model.fit_transform(self.X, self.y)

    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self, save_path:str):
        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)

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
        choices=['truthfulness_score' ,'informative' ,'consistency'],
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

def main(model:str=None, dataset:str=None,all_metrics:bool=False,metric:str=None,
         y_min:float=None, y_max:float=None):
    out_dir = os.path.join("saved_models","lookups",f"{model}-{dataset}")
    print(f"Making output directory at {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    eval_results = pd.read_json(f"./results/{model}-{dataset}/model_evaluations.jsonl",lines=True)
    print(f"Model rated itself with the following unique values {pd.unique(eval_results['score'])}")

    if(metric==None and all_metrics==False):
        raise ValueError("Either --metric or --all-metrics must be specified.")
    
    if(all_metrics):
        print("Creating RegressionTrainer")
        trainer = RegressionTrainer(eval_results, y_min=y_min, y_max=y_max)
        for metric in all_metric_list:
            trainer.set_metric(metric)
            trainer.train()
            score_dict = {}
            for i in range(6):
                pred = trainer.predict([i])
                score_dict[i] = pred[0]
            outfile = os.path.join(out_dir, f"{metric}.json")
            with open(outfile, "w") as f:
                json.dump(score_dict, f, indent=4)
            print(f"Saved regression model for metric {metric} at {outfile}")
    else:
        print("Creating RegressionTrainer")
        trainer = RegressionTrainer(eval_results, y_min=y_min, y_max=y_max)
        trainer.set_metric(metric)
        trainer.train()
        score_dict = {}
        for i in range(6):
            pred = trainer.predict([i])
            score_dict[i] = pred[0]
        outfile = os.path.join(out_dir, f"{metric}.json")
        with open(outfile, "w") as f:
            json.dump(score_dict, f, indent=4)
        print(f"Saved regression model for metric {metric} at {outfile}")

if __name__ == "__main__":
    args = parser()
    main(model=args.model, dataset=args.dataset, all_metrics=args.all_metrics, metric=args.metric,
         y_min=args.y_min, y_max=args.y_max)