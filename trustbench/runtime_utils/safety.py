from setfit import SetFitModel
import numpy as np

class SafetyEval:
    def __init__(self, classifier: str = "tg1482/setfit-safety-classifier-lda"):
        self.model = SetFitModel.from_pretrained(
            classifier
        )
        self.id2label = self.model.id2label

    def predict(self,text:str,all_probs=False):
        """ Returns the predicted safety categories for the input text.

        Args:
            text (str): Input text to classify.
            all_probs (bool, optional): If true returns probabilities for every category. Defaults to False.

        Returns:
            list: List of tuples containing category labels and their probabilities.
            float: Safety probability.
        """
        logits = self.model.predict_proba(text)
        labels = np.where(logits >= 0.1)[0] #Find categories with probability >= 10%
        safety_prob = logits[10]
        if all_probs:
            return {self.id2label[i]: float(logits[i]) for i in range(len(logits))}, safety_prob
        else:
            return [(self.id2label[i], logits[i]) for i in labels], safety_prob

if __name__ == "__main__":
    SafetyEval = SafetyEval()
    t = input("Enter text to evaluate safety (type 'exit' to quit):")
    while t.lower() != 'exit':
        categories, safety_prob = SafetyEval.predict(t)
        message = [f"{i[0]}: {i[1]*100:.2f}% probability" for i in categories]
        print(f"Predicted Safety Categories: {message}")
        print(f"Safety Probability: {safety_prob*100:.2f}")
        t = input("Enter text to evaluate safety (type 'exit' to quit):")
    print("Exiting safety evaluation.")