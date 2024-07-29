import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

from loguru import logger
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

from .base import BaseVectorStore
from .node import TextNode, VectorStoreQueryResult

logger.add(
    sink=sys.stdout,
    colorize=True,
    format="<green>{time}</green> <level>{message}</level>",
)

class RRF(BaseVectorStore):
    """Semantic Vector Store using SentenceTransformer embeddings."""

    saved_file: str = "rag-foundation/data/test_db_00.csv"
    embed_model_name: str = "all-MiniLM-L6-v2"
    embed_model: SentenceTransformer = SentenceTransformer(embed_model_name)

    def __init__(self, **data):
        super().__init__(**data)
        self._setup_store()
    
    def get(self, text_id: str) -> TextNode:
        """Get node."""
        try:
            return self.node_dict[text_id]
        except KeyError:
            logger.error(f"Node with id `{text_id}` not found.")
            return None

    def add(self, nodes: List[TextNode]) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            if node.embedding is None:
                logger.info(
                    "Found node without embedding, calculating "
                    f"embedding with model {self.embed_model_name}"
                )
                node.embedding = self._get_text_embedding(node.text)
            self.node_dict[node.id_] = node
        self._update_csv()  # Update CSV after adding nodes
        return [node.id_ for node in nodes]

    def delete(self, node_id: str, **delete_kwargs: Dict) -> None:
        """Delete nodes using node_id."""
        if node_id in self.node_dict:
            del self.node_dict[node_id]
            self._update_csv()  # Update CSV after deleting nodes
        else:
            logger.error(f"Node with id `{node_id}` not found.")
    

    def query(self, query: str, top_k: int = 3) -> VectorStoreQueryResult:
        doc_ids = list(self.node_dict.keys())
        passages = [node.text for node in self.node_dict.values()]
        ## cosine_score
        document_embeddings = self.embed_model.encode(passages, convert_to_tensor=True)
        query_embedding = self.embed_model.encode(query, convert_to_tensor=True)
        document_embeddings = document_embeddings.cuda()
        query_embedding = query_embedding.cuda()
        cosine_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]
        bert_ranking = {idx: rank + 1 for rank, idx in enumerate(np.argsort(cosine_scores.cpu().numpy())[::-1])}
        ## bm25 score
        bm25 = BM25Okapi([doc.split() for doc in passages])
        bm25_scores = bm25.get_scores(query.split())
        bm25_ranking = {idx: rank + 1 for rank, idx in enumerate(np.argsort(bm25_scores)[::-1])}
        ## tf-idf
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(passages)
        query_vec = tfidf_vectorizer.transform([query])
        tfidf_scores = np.dot(tfidf_matrix, query_vec.T).toarray().flatten()
        tfidf_ranking = {idx: rank + 1 for rank, idx in enumerate(np.argsort(tfidf_scores)[::-1])}
        ## RRF
        ranked_lists = [bm25_ranking, tfidf_ranking, bert_ranking]
        combined_scores = {}
        for ranked_list in ranked_lists:
            for doc, rank in ranked_list.items():
                combined_scores[doc] = combined_scores.get(doc, 0) + 1 / (60 + rank)
        combined_ranking = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        node_ids = [doc_ids[idx] for idx, _ in combined_ranking][:top_k]
        ##
        if len(passages) == 0:
            logger.error("No documents found in the index.")
            result_nodes, similarities, node_ids = [], [], []
        else:
            result_nodes = [self.node_dict[node_id] for node_id in node_ids]
            similarities = []
        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )
    
    def _get_text_embedding(self, text: str) -> List[float]:
        """Calculate embedding."""
        return self.embed_model.encode(text).tolist()
    
    def batch_query(
        self, query: List[str], top_k: int = 3
    ) -> List[VectorStoreQueryResult]:
        """Batch query similar nodes."""
        return [self.query(q, top_k) for q in query]
     