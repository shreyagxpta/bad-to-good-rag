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


# Embed the whole corpus once at init so each incoming query only embeds
# itself, instead of re-embedding every document per request. In prod, these
# vectors would live in a vectordb rather than an in-memory array.
doc_vecs = embed(docs)

# actual similarity checking

def cosine(query_vec, doc_matrix):
    """Cosine similarity between one query vector and every row of a matrix.

    Args:
        query_vec: np.ndarray of shape (embedding_dim).
        doc_matrix: np.ndarray of shape (n_docs, embedding_dim).

    Returns:
        np.ndarray of shape (n_docs,), the query's similarity to each row.
    """
    # Dividing by the vector lengths cancels magnitude, so the score reflects
    # direction (semantic similarity) rather than how long either vector is
    # e.g if query is [3,4] and 2 docs are [[3,4],[6,8]], normalizing will give dots [1,1] instead of [25,50]
    return (doc_matrix @ query_vec) / (np.linalg.norm(doc_matrix, axis=1) * np.linalg.norm(query_vec))


def answer(question):
    """Answer a question from the indexed corpus and print the result.

    Embeds the question, retrieves the most similar document, prompts the model
    to answer from it, and prints the question, the retrieved source, and the
    generated answer.

    Args:
        question: the user's question as a string.
    """
    q_vec = embed([question])[0]      # embed the question with the same model as the docs
    sims = cosine(q_vec, doc_vecs)    # similarity of the question to each document

    # Retrieval keeps only the single highest-scoring document (top-k = 1). If the
    # answer is spread across two documents, the second is never retrieved and the
    # model cannot see it. good_rag.py will retrieve the top k and pass them all.
    top = int(np.argmax(sims))
    context = docs[top]

    # The prompt does not restrict the model to the provided context or invite it
    # to say it does not know. So when retrieval misses, the model answers from its
    # own training data instead of flagging the gap. good_rag.py adds a grounded
    # instruction telling the model to use only the context and abstain otherwise.
    # this shows the importance of prompt engineering!!

    prompt = f"Answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    print(f"\nQ: {question}")
    print(f"[retrieved: {paths[top]}]")  # which document was retrieved: the key retrieval signal to debug
    print(f"A: {response.choices[0].message.content}")


if __name__ == "__main__":
    # Each question below exercises one of the limitations documented above.

    # The answer (ratio and temperature) is split across two documents, so top-k = 1
    # retrieval can surface at most one of the two pieces.
    answer("What coffee-to-water ratio and water temperature should I use for a strong pour-over?")

    # Cold brew is not covered anywhere in the corpus, so this tests whether the
    # model abstains or fabricates when retrieval has nothing relevant to return.
    answer("What water temperature is best for cold brew?")