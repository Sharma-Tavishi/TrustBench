import os
import json
import glob

class ModelConfidenceMapper:
    def __init__(self, base_dir="saved_models/lookups"):
        self.base_dir = base_dir

    def set_model_dataset(self, model_name: str, dataset: str):
        model_dir = os.path.join(self.base_dir,f"{model_name}-{dataset}")

        json_paths = glob.glob(os.path.join(model_dir,"*.json"))
        jsons = [os.path.split(i)[-1][:-5] for i in json_paths]
        
        self.score_to_dict = {}
        for i,file in enumerate(json_paths):
            with open(file) as f:
                data = json.loads(f.read().replace("NaN", "null"))
            self.score_to_dict[jsons[i]] = data
    
    def get_metric(self, score: int, metric_name: str):
        score = str(score)
        if metric_name in self.score_to_dict:
            return self.score_to_dict[metric_name][score]
        else:
            raise ValueError(f"Metric {metric_name} not found.")
    
    def get_all_metrics(self, score: int):
        score = str(score)
        metrics = {}
        for k,v in self.score_to_dict.items():
            metrics[k] = v[score]
        return metrics


if __name__ == "__main__":
    # data = map_model_confidence_scores("saved_models/lookups")
    # model = input("Enter model name:")
    # dataset = input("Enter dataset name:")
    model= "llama3:8b"
    dataset="mixed_qa"
    score = 3
    cm = ModelConfidenceMapper()
    cm.set_model_dataset(model,dataset)
    metrics = cm.get_all_metrics(score)
    print(metrics)