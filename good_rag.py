"""Improved RAG over a small document corpus.

The improved counterpart to bad_rag.py. It fixes the baseline's flaws one lever
at a time, so each change can be seen in isolation. Levers, in order: grounding
(a prompt fix so the model answers only from context and abstains otherwise),
then top-k retrieval instead of a single doc, then chunking (a no-op on this
tiny corpus, so a longer doc gets added to show it), then reranking.

Usage:
    python good_rag.py

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

# Grounding instruction. Kept in the system message so the model treats it as a
# binding rule rather than one suggestion among the data it's given.
SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer the user's question using only "
    "the provided context. If the context does not contain the answer, say you don't "
    "know instead of guessing or drawing on outside knowledge. Do not add information "
    "that is not in the context."
)

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
    to answer from it under grounding instruction, and prints the question, the retrieved source, and the
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

    # Grounding: the rule to use only the context, and to abstain when the answer
    # is not present, lives in the system message (SYSTEM_PROMPT). The user message
    # carries only the data (the retrieved context and the question). This is the
    # change from bad_rag.py, and it stops the model from filling gaps
    # with its own training data.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
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