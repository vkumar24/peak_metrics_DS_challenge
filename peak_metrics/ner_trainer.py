import spacy
import warnings
nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings("ignore")

class NerTrainer():
    def __init__(self):
        pass

    def extract_airline_brands(self, text):
        """ assign topics to text"""
        doc = nlp(text)
        airline_brands = []
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                airline_brands.append(ent.text)
        return airline_brands

    def __call__(self, text):
        airline_brands = self.extract_airline_brands(text)
        return airline_brands