"""Improved RAG over a small document corpus.

The improved counterpart to bad_rag.py. It fixes the baseline's flaws one lever
at a time, so each change can be seen in isolation. Levers, in order: grounding
(a prompt fix so the model answers only from context and abstains otherwise),
then top-k retrieval instead of a single doc, then chunking (a no-op on this
tiny corpus, so a longer doc - brewing guide- gets added to show it), then reranking.

Usage:
    python good_rag.py

Requires OPENAI_API_KEY in a local .env file.
"""

import glob
import re

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

RERANK_PROMPT = (
    "You are a relevance grader. Given a question and a passage, rate how well the "
    "passage answers the question, from 0 (irrelevant) to 10 (directly and completely "
    "answers it). Respond with only the number."
)

# First-stage retrieval pulls a wide candidate set by embedding similarity (cheap
# but coarse); reranking then re-scores those candidates and keeps the best.
RETRIEVE_K = 6   # candidates fetched by embedding similarity, before reranking
TOP_K = 3        # chunks kept after reranking and sent to the generator

# --- Indexing (runs once at startup) --------------------------------------
# Load the corpus and turn it into embedding vectors that can be searched by
# semantic similarity (via cosine, linalg101) when a query arrives.

def split_into_chunks(text, min_chars=30):
    """Split a document into paragraph-sized chunks on blank lines.

    Drops whitespace-only or very short fragments (like bare headings) that
    carry no real signal. Prod splitters are usually more involved
    (fixed-size windows with overlap, or structure-aware splitting), but blank-line
    paragraphs are a clear, simple unit for this corpus.

    Args:
        text: the full document text.
        min_chars: minimum length for a fragment to be kept as a chunk.

    Returns:
        list of chunk strings.
    """
    pieces = [c.strip() for c in text.split("\n\n")]
    return [c for c in pieces if len(c) >= min_chars]

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

# Index at the CHUNK level rather than the whole file. Each chunk records the file
# and position it came from ("path#i"), so retrieval can report exactly which slice
# it used. A long multi-topic document now contributes one vector per section
# instead of a single blurred vector for the whole file.
paths = sorted(glob.glob("docs/*.md"))
chunks = []
chunk_sources = []
for path in paths:
    for i, chunk in enumerate(split_into_chunks(open(path).read())):
        chunks.append(chunk)
        chunk_sources.append(f"{path}#{i}")
chunk_vecs = embed(chunks)

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

def _parse_score(text):
    """Extract the first number from a reranker response; default to 0.0 if none."""
    match = re.search(r"\d+(?:\.\d+)?", text)
    return float(match.group()) if match else 0.0


def rerank(question, candidate_idx):
    """Re-score candidate chunks by how well each answers the question.

    Embedding similarity is coarse: it rewards surface overlap, so a chunk about
    "water temperature" can outrank the chunk that actually answers "what kind of
    water." This pass shows the question and each chunk to the model together and
    asks for a direct relevance score, then sorts by it. In prod, this stage is
    a purpose-built cross-encoder or reranking model (e.g. Cohere Rerank), which is
    far cheaper than one LLM call per candidate; the LLM stands in here to avoid
    adding a dependency.

    Args:
        question: the user's question.
        candidate_idx: chunk indices from the first-stage embedding retrieval.

    Returns:
        list of (chunk_index, score) tuples sorted by score, highest first.
    """
    scored = []
    for i in candidate_idx:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": RERANK_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nPassage: {chunks[i]}"},
            ],
        )
        scored.append((int(i), _parse_score(response.choices[0].message.content)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored

def answer(question):
    """Answer a question from the indexed corpus and print the result.

    Retrieves a wide candidate set by embedding similarity, reranks it, keeps the
    top TOP_K chunks, and prompts the model to answer from them under a grounding
    instruction. Prints the question, the final chunks with their rerank scores,
    and the generated answer.

    Args:
        question: the user's question as a string.
    """
    q_vec = embed([question])[0]      # embed the question with the same model as the chunks
    sims = cosine(q_vec, chunk_vecs)    # similarity of the question to each chunk

    # Stage 1 - retrieve a wide candidate set by embedding similarity (cheap, coarse).
    candidate_idx = np.argsort(sims)[::-1][:RETRIEVE_K]

    # Stage 2 - rerank those candidates and keep the top TOP_K for the answer.
    ranked = rerank(question, candidate_idx)[:TOP_K]
    context = "\n\n".join(chunks[i] for i, _ in ranked)

    # Grounding: the rule to use only the context, and to abstain when the answer
    # is not present, lives in the system message (SYSTEM_PROMPT). The user message
    # carries only the data (the retrieved context and the question).
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )

    print(f"\nQ: {question}")
    retrieved = ", ".join(f"{chunk_sources[i]} ({score:.0f})" for i, score in ranked)
    print(f"[retrieved: {retrieved}]")  # final chunks with rerank scores, best first
    print(f"A: {response.choices[0].message.content}")


if __name__ == "__main__":
    # Each question below exercises one of the limitations documented above.

    # The answer (ratio and temperature) is split across two documents, so top-k = 1
    # retrieval can surface at most one of the two pieces.
    answer("What coffee-to-water ratio and water temperature should I use for a strong pour-over?")

    # Cold brew is not covered anywhere in the corpus, so this tests whether the
    # model abstains or fabricates when retrieval has nothing relevant to return.
    answer("What water temperature is best for cold brew?")

    # Only the brewing guide covers water quality. With chunking, this should
    # retrieve just that section rather than dragging in the whole guide.
    answer("What kind of water should I use for brewing?")