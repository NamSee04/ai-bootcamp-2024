import torch
from transformers import BertTokenizer, BertModel
import numpy as np

class BERTScore:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
    
    def tokenize_texts(self, texts):
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        return encoded_input

    def extract_embeddings(self, encoded_input):
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        return outputs.last_hidden_state

    def compute_similarity(self, embeddings1, embeddings2):
        embeddings1 = embeddings1.squeeze(0).cpu().numpy()
        embeddings2 = embeddings2.squeeze(0).cpu().numpy()
        dot_product = np.dot(embeddings1, embeddings2.T)
        magnitude1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        magnitude2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        similarities = dot_product / np.dot(magnitude1, magnitude2.T)
        return similarities

    def calculate_bertscore(self, reference_text, candidate_text):
        encoded_ref = self.tokenize_texts([reference_text])
        encoded_cand = self.tokenize_texts([candidate_text])
        ref_embeddings = self.extract_embeddings(encoded_ref)
        cand_embeddings = self.extract_embeddings(encoded_cand)
        similarities = self.compute_similarity(ref_embeddings, cand_embeddings)
        precision = similarities.max(axis=1).mean()
        recall = similarities.max(axis=0).mean()
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
        