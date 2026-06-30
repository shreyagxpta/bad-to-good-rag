"""Minimal RAG baseline over a small coffee-based document corpus :) bc I like coffee.

This is an intentionally simple reference implementation. It embeds each document,
retrieves the single closest one to a query, and asks an LLM to answer from it. It
favors readability over completeness and deliberately omits several production
concerns (document chunking, multi-document retrieval, and grounded prompting) that
the companion good_rag.py will aim to address. It exists as a baseline for seeing what each of
those levers contributes.

Usage:
    python bad_rag.py

Requires OPENAI_API_KEY in a local .env file.
"""

import glob

import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load variables from .env into the environment (notably OPENAI_API_KEY).
# The OpenAI client reads the key from the environment automatically.
load_dotenv()
client = OpenAI()


# --- Indexing (runs once at startup) --------------------------------------
# Load the corpus and turn it into embedding vectors that can be searched by
# semantic similarity (via cosine, linalg101) when a query arrives.

# Each md file is treated as a single document, so one embedding has to
# represent the whole file. When a file covers several topics, that vector
# becomes a blurred average of them and retrieval gets less precise. good_rag.py
# will split documents into smaller, single-topic chunks to avoid this.
paths = sorted(glob.glob("docs/*.md"))
docs = [open(p).read() for p in paths]


def embed(texts):
    """Embed a list of strings into vectors using OpenAI's embedding model.

    Args:
        texts: list of strings to embed.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), one row vector per input.
    """
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([item.embedding for item in response.data])


# Embed the whole corpus once at startup so each incoming query only embeds
# itself, instead of re-embedding every document per request. In production these
# vectors would live in a vector database rather than an in-memory array.
doc_vecs = embed(docs)