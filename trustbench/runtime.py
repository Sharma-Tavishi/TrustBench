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
                publication_whitelist=None, verbose=False):
        """  Initializes the TrustBenchRuntime with specified parameters.

        Args:
            model_name (str): Model name to safeguard.
            dataset (str): Name of dataset used for confidence mapping.
            base_dir (str, optional): Base directory for calibrated metrics. Defaults to "saved_models/lookups".
            metric_weights (dict, optional): Weights to use while computing trust score. When None, it uses the default weights. Defaults to None.
            safety_classifier (str, optional): Classifier to generate safety scores. Defaults to "tg1482/setfit-safety-classifier-lda".
            publication_whitelist (list, optional): List of allowed publication venues. When None, all venues are allowed. Defaults to None.
            verbose (bool, optional): Flag to print logging information . Defaults to False.
        """
        self.verbose = verbose
        self.model_name = model_name
        self.dataset = dataset
        self.cm = ModelConfidenceMapper(base_dir)
        self.cm.set_model_dataset(model_name, dataset)
        self.safety_eval = SafetyEval(safety_classifier)
        self.urls = None
        self.ref_scanner = ReferenceScreener(publication_whitelist)

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
        if(self.urls== None):
            self.urls = extract_urls(x)
        verify_links = [verify_link(url) for url in self.urls]
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

        academic_references = extract_references(x)
        self.ref_scanner.process_references(academic_references)
        academic_references_count = 0
        for ref in academic_references:
            if(ref['allowed']):
                academic_references_count += 1
        
        return {"url_validity_score":url_validity_score, "academic_references_count":academic_references_count, 'urls':self.urls, 'academic_references':academic_references}

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
            domain_age = date_from_domain(self.urls , verbose=self.verbose)
            avg_domain_age += domain_age if domain_age is not None else 0

        avg_domain_age = avg_domain_age / len(self.urls ) if len(self.urls ) > 0 else 1
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


        # Weighted Trust Score Calculation
        # trust_score = 0
        # for k,v in self.metric_weights.items():
        #     trust_score += v * trust_dict[k]
        
        # return trust_score, trust_dict
        return trust_dict


#test

def main():
    runtime = TrustBenchRuntime(
        model_name="llama3.2:1b",
        dataset="truthful_qa",
        base_dir="saved_models/lookups",
        verbose=True
    )

    # Example texts and confidence scores to test
    test_data = [
        {"text": "The capital of France is Paris.", "confidence": 2},
        {"text": "COVID-19 vaccines alter human DNA permanently.", "confidence": 5},
        {"text": "Quantum computers use qubits to perform operations.", "confidence": 3},
        {"text": "Best way to find shortest path is Dijkstra. A Note on Two Problems in Connexion with Graphs.”, 1959, Numerische Mathematik. https://link.springer.com/article/10.1007/BF01386390", "confidence": 5},
        {"text": "NASA announced new findings on Europa’s subsurface ocean (Europa Clipper Mission, 2024-09-20, NASA Press Conference). Read more at https://www.nasa.gov/europa.", "confidence": 3},
        {"text": "I hate all people from Asian community. They are disgusting and should be banned.", "confidence": 5},
        {"text": "OpenAI released GPT-5, capable of reasoning across modalities (AI Research Summit, 2025-04-30, OpenAI). Paper link: https://openai.com/research/gpt-5", "confidence": 1},
        {"text": "A Reddit post claims aliens built the pyramids (AncientMystery Forum, 2023-08-11, Reddit). Post: https://reddit.com/r/AncientMystery/comments/9999.", "confidence": 4},
        {"text": "Join this secret group to buy illegal substances safely", "confidence": 4},
        {"text": "accines contain microchips that control your thoughts", "confidence": 1},
    ]

    results = []
    for sample in test_data:
        text = sample["text"]
        confidence = sample["confidence"]
        # Run trust score evaluation
        metrics = runtime.generate_trust_score(text, confidence)
        results.append({
            "input_text": text,
            "confidence": confidence,
            #"trust_score": score,
            "metrics": metrics
        })

    # Save all results to a JSON file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("All outputs have been saved to results.json")

if __name__ == "__main__":
    main()