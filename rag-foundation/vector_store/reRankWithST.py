import sys
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

from loguru import logger
from sentence_transformers import SentenceTransformer, CrossEncoder, util

from .base import BaseVectorStore
from .node import TextNode, VectorStoreQueryResult

logger.add(
    sink=sys.stdout,
    colorize=True,
    format="<green>{time}</green> <level>{message}</level>",
)

class reRankWithST(BaseVectorStore):
    """Semantic Vector Store using SentenceTransformer embeddings."""

    saved_file: str = "rag-foundation/data/test_db_00.csv"
    embed_model_name: str = "all-MiniLM-L6-v2"
    embed_model: SentenceTransformer = SentenceTransformer(embed_model_name)
    croos_model: CrossEncoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

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
        corpus_embeddings = [node.embedding for node in self.node_dict.values()]
        ## sematic_score
        question_embedding = self.embed_model.encode(query, convert_to_tensor=True)
        question_embedding = question_embedding.cuda()
        hits = util.semantic_search(question_embedding, torch.tensor(corpus_embeddings), top_k=top_k)
        hits = hits[0]
        ## rerank
        cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
        cross_scores = self.croos_model.predict(cross_inp)
        # Sort results by the cross-encoder scores
        for idx in range(len(cross_scores)):
            hits[idx]['cross-score'] = cross_scores[idx]
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        node_ids = [doc_ids[mp['corpus_id']] for mp in hits]
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