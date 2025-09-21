"""
MRtrix3 Knowledge Base Population Package

This package contains modules for populating the MRtrix3 RAG database:
- populate_database: Main ETL pipeline for documentation
- generate_embeddings: Vector embedding generation for semantic search
- error_extractor: Extract error messages from source code
"""

from .error_extractor import ErrorExtractor
from .generate_embeddings import EmbeddingGenerator

__all__ = ["ErrorExtractor", "EmbeddingGenerator"]
