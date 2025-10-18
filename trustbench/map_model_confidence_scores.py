import os
import json
import glob

def get_dict(model_name:str,dataset:str):
    base_dir="saved_models/lookups"
    model_dir = os.path.join(base_dir,f"{model_name}-{dataset}")

    json_paths = glob.glob(os.path.join(model_dir,"*.json"))
    jsons = [os.path.split(i)[-1][:-5] for i in json_paths]
    score_to_dict = {}
    for i,file in enumerate(json_paths):
        with open(file) as f:
            data = json.loads(f.read().replace("NaN", "null"))
        score_to_dict[jsons[i]] = data
    return score_to_dict

def get_metrics_from_score(model:str,dataset:str,score:int):
    score_to_dict = get_dict(model,dataset)
    metrics = {}
    score = str(score)
    for k,v in score_to_dict.items():
        metrics[k] = v[score]
    return metrics

def map_model_confidence_scores(base_dir="saved_models/lookups"):
    """
    Builds a dictionary mapping:
        {
          model_name: {
              confidence_level: {
                  metric_name: score
              }
          }
        }
    Example:
        {
          "llama3:8b-mixed_qa": {
              1: {"bertscore_f1": 0.79, "f1": 0.76},
              2: {"bertscore_f1": 0.80, "f1": 0.77},
              ...
          }
        }
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Could not find directory: {base_dir}")

    results = {}

    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        # Collect per-confidence data
        conf_map = {i: {} for i in range(6)}  # confidence 0–5

        for file_name in os.listdir(model_path):
            if not file_name.endswith(".json"):
                continue

            metric = file_name[:-5]  # strip ".json"
            with open(os.path.join(model_path, file_name)) as f:
                data = json.loads(f.read().replace("NaN", "null"))

            for conf_str, score in data.items():
                conf = int(conf_str)
                conf_map[conf][metric] = score

        results[model_name] = conf_map

    return results


if __name__ == "__main__":
    # data = map_model_confidence_scores("saved_models/lookups")
    # model = input("Enter model name:")
    # dataset = input("Enter dataset name:")
    model= "llama3:8b"
    dataset="mixed_qa"
    score = 3
    print(get_metrics_from_score(model,dataset,score))
