from runtime_utils.confidence import ModelConfidenceMapper
from runtime_utils.citation import *
from runtime_utils.safety import SafetyEval
from runtime_utils.timeliness import date_from_domain

from datetime import datetime
import numpy as np
import tqdm
import json
import re


class TrustBenchRuntime:
    def __init__(self, model_name: str, dataset: str, 
                base_dir="saved_models/lookups", 
                metric_weights :dict = None,
                safety_classifier: str = "tg1482/setfit-safety-classifier-lda",
                publication_whitelist=None, verbose=False,
                link_verify=True, check_link_time=True):
        """  Initializes the TrustBenchRuntime with specified parameters.

        Args:
            model_name (str): Model name to safeguard.
            dataset (str): Name of dataset used for confidence mapping.
            base_dir (str, optional): Base directory for calibrated metrics. Defaults to "saved_models/lookups".
            metric_weights (dict, optional): Weights to use while computing trust score. When None, it uses the default weights. Defaults to None.
            safety_classifier (str, optional): Classifier to generate safety scores. Defaults to "tg1482/setfit-safety-classifier-lda".
            publication_whitelist (list, optional): List of allowed publication venues. When None, all venues are allowed. Defaults to None.
            verbose (bool, optional): Flag to print logging information . Defaults to False.
            link_verify (bool, optional): Flag to verify links in citation score. Defaults to True.
            check_link_time (bool, optional): Flag to set time limit while checking links. Defaults to True.
        """
        self.verbose = verbose
        self.model_name = model_name
        self.dataset = dataset
        self.cm = ModelConfidenceMapper(base_dir)
        self.cm.set_model_dataset(model_name, dataset)
        self.safety_eval = SafetyEval(safety_classifier)
        self.ref_scanner = ReferenceScreener(publication_whitelist)
        ## Class variables to store intermediate results
        self.academic_references = None
        self.urls = None
        self.link_verify = link_verify
        self.check_link_time = check_link_time

        ## Need to update these to make it a better default
        if(metric_weights is None):
            self.load_metric_weights(None)
        else:
            self.metric_weights = metric_weights
    
    def load_metric_weights(self, json_path: str=None):
        """ Loads metric weights from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing metric weights. When None loads based on model and dataset name. Defaults to None.
        """
        if(json_path is None):
            json_path = f"saved_models/runtime_weights/{self.model_name}-{self.dataset}.json"
        with open(json_path) as f:
            self.metric_weights = json.loads(f.read().replace("NaN", "null"))
    
    def citation_score(self, x:str) -> dict:
        """ Generates citation score for the given input text.
        Args:
            x (str): Input text to evaluate.
        Returns:
            dict: Dictionary containing citation metrics - 'url_validity_score' and 'academic_references_count'.
        """
        if(self.verbose):
            print("Extracting url sources...")
        if(self.urls== None):
            self.urls = extract_urls(x)
        if(self.link_verify):
            verify_links = [verify_link(url) for url in self.urls]
        else:
            verify_links = [True for url in self.urls]
        url_validity_score = 0
        if(self.verbose):
            print("Verifying urls...")
            for i in tqdm.tqdm(range(len(self.urls ))):
                url_validity_score += int(verify_links[i])
        else:
            for i in range(len(self.urls )):
                url_validity_score += int(verify_links[i])
        
        url_validity_score = url_validity_score / len(self.urls ) if len(self.urls ) > 0 else 1 

        if(self.verbose):
            print("Extracting academic references...")

        if(self.academic_references is None):
            self.academic_references = extract_references(x)
        
        self.ref_scanner.process_references(self.academic_references)
        academic_references_count = 0
        for ref in self.academic_references:
            if(ref['allowed']):
                academic_references_count += 1
        
        return {"url_validity_score":url_validity_score, "academic_references_count":academic_references_count, 'urls':self.urls, 'academic_references':self.academic_references}

    def get_metrics_from_score(self, score: int) -> dict:
        """ Generates all metrics from the given confidence score.
        Args:
            score (int): self-reported confidence score.

        Returns:
            dict: Dictionary containing all metrics corresponding to the confidence score.
        ---
        Metrics returned 
        * 'metrics': ['rouge_l', 'f1','bertscore_f1'],
        * 'nli': ['nli_entailment', 'nli_contradiction', 'nli_neutral'],
        * 'fconsistency': ['ng1_prec','ng1_rec','ng1_f1']
        """
        return self.cm.get_all_metrics(score)

    def safety_score(self, x:str)-> dict:
        """ Generates safety score for the given input text.

        Args:
            x (str): Input text to evaluate.

        Returns:
            dict: Dictionary containing safety categories with over 10% probability and safety probability.
        """
        categories, safety_prob = self.safety_eval.predict(x)
        if(self.verbose):
            print(f"Predicted Safety Categories: {categories}")
            print(f"Safety Probability: {safety_prob*100:.2f}%")

        return {"safety_categories": categories, "safety_probability": safety_prob}

    def timeliness_score(self, x:str) -> dict:
        """ Generates timeliness score based on domain age of URLs in the input text.
        Args:
            x (str): Input text to evaluate.
        Returns:
            dict: Dictionary containing 'average_domain_age' in years.
        """
        if(self.urls== None):
            self.urls = extract_urls(x)
        avg_domain_age = 0
        for url in self.urls :
            if(self.check_link_time):
                domain_age = date_from_domain(url , verbose=self.verbose)
            else:
                domain_age = 0
            avg_domain_age += domain_age if domain_age is not None else 0

        avg_domain_age = avg_domain_age / len(self.urls ) if len(self.urls ) > 0 else 1
        if(self.verbose):
            print(f"Average Domain Age Since Last Updated: {avg_domain_age} years")
        
        if(self.verbose):
            print(f"Checking academic references timeliness...")
        if(self.academic_references is None):
            self.academic_references = extract_references(x)

        current_year = datetime.now().year
        avg_reference_age = 0
        total = 0
        for ref in self.academic_references:
            if ref['allowed']:
                avg_reference_age += (current_year - int(ref['year']))
                total += 1
        avg_reference_age = avg_reference_age / total if total > 0 else 1
        if(self.verbose):
            print(f"Average Academic Reference Age from allowed venues: {avg_reference_age} years")

        return {"average_domain_age": 1/(avg_domain_age+1), "average_reference_age": 1/(avg_reference_age+1)}

    def generate_trust_score(self, x:str, score:int) -> tuple[float, dict]:
        """ Generates the overall trust score along with individual metric scores.
        Args:
            x (str): Input text to evaluate.
            score (int): self-reported confidence score.

        Returns:
            tuple[float, dict]: Overall trust score and dictionary of individual metric scores.
        """
        trust_dict = {}
        self.urls = None

        if(self.verbose):
            print("Generating Safety Score...")
        trust_dict.update(self.safety_score(x))

        if(self.verbose):
            print("Generating Metrics from Score...")
        trust_dict.update(self.get_metrics_from_score(score))
        
        if(self.verbose):
            print("Generating Citation Score...")
        trust_dict.update(self.citation_score(x))

        if(self.verbose):
            print("Generating Timeliness Score...")
        trust_dict.update(self.timeliness_score(x)) #removed verbose

        if trust_dict['safety_probability']<=0.6:
            if(self.verbose):
                print("Safety probability below threshold. Setting trust score to 0.")
            return 0.0, trust_dict

        # Weighted Trust Score Calculation
        trust_score = 0
        for k,v in self.metric_weights.items():
            trust_score += float(v) * trust_dict[k]
        
        return trust_score, trust_dict
        # return trust_dict


#test

# def main():
#     runtime = TrustBenchRuntime(
#         model_name="llama3.2:1b",
#         dataset="truthful_qa",
#         base_dir="saved_models/lookups",
#         verbose=True
#     )

#     # Example texts and confidence scores to test
#     test_data = [
        
#     ]

#     results = []
#     for sample in test_data:
#         text = sample["text"]
#         confidence = sample["confidence"]
#         # Run trust score evaluation
#         metrics = runtime.generate_trust_score(text, confidence)
#         results.append({
#             "input_text": text,
#             "confidence": confidence,
#             #"trust_score": score,
#             "metrics": metrics
#         })

#     # Save all results to a JSON file
#     with open("results.json", "w") as f:
#         json.dump(results, f, indent=4)

#     print("All outputs have been saved to results.json")

# if __name__ == "__main__":
#     main()