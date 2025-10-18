from runtime_utils.confidence import ModelConfidenceMapper
from runtime_utils.citation import *
from runtime_utils.safety import SafetyEval
from runtime_utils.timeliness import date_from_domain

import numpy as np
import tqdm
import json

class TrustBenchRuntime:
    def __init__(self, model_name: str, dataset: str, 
                base_dir="saved_models/lookups", 
                metric_weights :dict = None,
                safety_classifier: str = "tg1482/setfit-safety-classifier-lda",
                verbose=False):
        """  Initializes the TrustBenchRuntime with specified parameters.

        Args:
            model_name (str): Model name to safeguard.
            dataset (str): Name of dataset used for confidence mapping.
            base_dir (str, optional): Base directory for calibrated metrics. Defaults to "saved_models/lookups".
            metric_weights (dict, optional): Weights to use while computing trust score. When None, it uses the default weights. Defaults to None.
            safety_classifier (str, optional): Classifier to generate safety scores. Defaults to "tg1482/setfit-safety-classifier-lda".
            verbose (bool, optional): Flag to print logging information . Defaults to False.
        """
        self.verbose = verbose
        self.model_name = model_name
        self.dataset = dataset
        self.cm = ModelConfidenceMapper(base_dir)
        self.cm.set_model_dataset(model_name, dataset)
        self.safety_eval = SafetyEval(safety_classifier)

        ## Need to update these to make it a better default
        if(metric_weights is None):
            self.metric_weights = {
                "confidence": 0.5,
                "citation": 0.3,
                "safety": 0.2
            }
        else:
            self.metric_weights = metric_weights
    
    def load_metric_weights(self, json_path: str):
        """ Loads metric weights from a JSON file.

        Args:
            json_path (str): Path to the JSON file containing metric weights.
        """
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
        urls = extract_urls(x)
        verify_links = [verify_link(url) for url in urls]
        url_validity_score = 0
        if(self.verbose):
            print("Verifying urls...")
            for i in tqdm(range(len(urls))):
                url_validity_score += int(verify_links[i])
        else:
            for i in range(len(urls)):
                url_validity_score += int(verify_links[i])
        
        url_validity_score = url_validity_score / len(urls) if len(urls) > 0 else 1 

        if(self.verbose):
            print("Extracting academic references...")

        academic_references = extract_references(x)
        return {"url_validity_score":url_validity_score, "academic_references_count":len(academic_references)}

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
        urls = extract_urls(x)
        avg_domain_age = 0
        for url in urls:
            domain_age = date_from_domain(url, verbose=self.verbose)
            avg_domain_age += domain_age if domain_age is not None else 0

        avg_domain_age = avg_domain_age / len(urls) if len(urls) > 0 else 1
        if(self.verbose):
            print(f"Average Domain Age Since Last Updated: {avg_domain_age} years")
        
        return {"average_domain_age": avg_domain_age}

    def generate_trust_score(self, x:str, score:int) -> tuple[float, dict]:
        """ Generates the overall trust score along with individual metric scores.
        Args:
            x (str): Input text to evaluate.
            score (int): self-reported confidence score.

        Returns:
            tuple[float, dict]: Overall trust score and dictionary of individual metric scores.
        """
        trust_dict = {}

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
        trust_dict.update(self.timeliness_score(x, verbose=self.verbose))

        # Weighted Trust Score Calculation
        trust_score = 0
        for k,v in self.metric_weights.items():
            trust_score += v * trust_dict[k]
        
        return trust_score, trust_dict