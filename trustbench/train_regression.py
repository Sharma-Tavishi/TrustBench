from sklearn.isotonic import IsotonicRegression
import pandas as pd
import numpy as np
import argparse

import os
import pickle
import json
import tqdm

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print(api_key)

all_metrics_dict = {'metrics': ['rouge_l', 'f1','bertscore_f1'],
               'nli': ['nli_entailment', 'nli_contradiction', 'nli_neutral'],
               'fconsistency': ['ng1_prec','ng1_rec','ng1_f1']}

class RegressionTrainer:
    def __init__(self, joined_data:pd.DataFrame, 
                 y_min=None, y_max=None):
        self.model = IsotonicRegression(y_min=y_min, y_max=y_max)

        self.joined_data = joined_data
        # Convert score to numeric, coerce errors to NaN, then fill NaN with 1
        self.joined_data ['score'] = pd.to_numeric(self.joined_data ['score'], errors='coerce').fillna(1)
        ## Set dummy values for X and y
        self.X = self.joined_data[['score']].values
        self.y = None
        self.y_pred = None

    def set_metric(self, metric_label:str):
        self.y = self.joined_data[metric_label].values

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
        choices=['rouge_l', 'f1','bertscore_f1', 'nli_entailment', 'nli_contradiction', 'nli_neutral','ng1_prec','ng1_rec','ng1_f1'],
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

def load_data(model_name, dataset_name, metric_group):
    file_path =f"./results/{model_name}-{dataset_name}/{metric_group}_detail.jsonl"
    metrics = pd.read_json(file_path, lines=True)
    return metrics

def get_joined_data(model_out:pd.DataFrame,metrics:pd.DataFrame) -> pd.DataFrame:
    return pd.merge(model_out, metrics, on='id').dropna()

def main(model:str=None, dataset:str=None,all_metrics:bool=False,metric:str=None,
         y_min:float=None, y_max:float=None):
    out_dir = os.path.join("saved_models","lookups",f"{model}-{dataset}")
    print(f"Making output directory at {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    model_outs = pd.read_json(f"./results/{model}-{dataset}/outputs_with_confidence.jsonl",lines=True)
    print(f"Model rated itself with the following unique values {pd.unique(model_outs['score'])}")

    if(metric==None and all_metrics==False):
        raise ValueError("Either --metric or --all-metrics must be specified.")
    
    if(all_metrics):
        for key,value in all_metrics_dict.items():
            print("Dataframe for metrics:", key)
            metrics_df = load_data(model, dataset, key)
            joint_df = get_joined_data(model_outs, metrics_df)
            trainer = RegressionTrainer(joint_df, y_min=y_min, y_max=y_max)
            for metric in value:
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
        found_k = None
        for k,v in all_metrics_dict.items():
            if metric in v:
                found_k = k
                break
        metrics_df = load_data(model, dataset, found_k)
        joint_df = get_joined_data(model_outs, metrics_df)
        trainer = RegressionTrainer(joint_df, y_min=y_min, y_max=y_max)
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