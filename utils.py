import re, pickle, faiss, numpy as np, google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ---------- configuration ----------
GENAI_KEY   = "AIzaSyAGV0V-TxzhLSct59Q_9KjZqbp-ADvpcXI"
INDEX_PATH  = "faiss_index_bge.idx"
META_PATH   = "qa_metadata_bge.pkl"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
TOP_K       = 8           # how many results to return
SIM_THRESHOLD = 0.50      # lower threshold for more matches
# -----------------------------------

genai.configure(api_key=GENAI_KEY)
embedder = SentenceTransformer(EMBED_MODEL)
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    qa_data = pickle.load(f)

_punct = re.compile(r"[^\w\s]")

def _normalize(text: str) -> str:
    """lowercase + strip punctuation + extra spaces"""
    # First convert to lowercase
    text = text.lower()
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def _semantic_search(query: str, debug=False):
    nq = _normalize(query)
    emb = embedder.encode([nq], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(emb, TOP_K)

    pairs = []
    if debug:
        print(f"\n[DEBUG] Semantic search for: '{query}' (normalized: '{nq}')")
    
    # First try exact match after normalization
    for qa in qa_data:
        if qa["normalized_question"] == nq:
            return [{
                "question": qa["question"],
                "answer": qa["answer"],
                "score": 1.0
            }]

    # If no exact match, use semantic search with very lenient threshold
    for score, i in zip(scores[0], idx[0]):
        q = qa_data[i]["question"]
        a = qa_data[i]["answer"]
        if debug:
            print(f"  Score: {score:.3f}  Q: {q}")
        pairs.append({
            "question": q,
            "answer": a,
            "score": float(score)
        })

    # Try semantic similarity with a very low threshold
    filtered = [p for p in pairs if p["score"] >= 0.05]  # Very low threshold
    if filtered:
        return filtered

    # Improved fallback: match whole words ignoring short stopwords
    fallback_matches = []
    query_words = set(w for w in nq.split() if len(w) > 2)  # ignore tiny words like 'is', 'in'

    for qa in qa_data:
        qnorm = qa["normalized_question"]
        q_words = set(qnorm.split())
        # Count matching words
        matching_words = len(query_words & q_words)
        if matching_words >= 1:  # Very lenient - just 1 matching word
            fallback_matches.append({
                "question": qa["question"],
                "answer": qa["answer"],
                "score": matching_words / len(query_words)  # Normalized score
            })

    # Sort fallback matches by score
    fallback_matches.sort(key=lambda x: x["score"], reverse=True)
    return fallback_matches[:TOP_K]


def get_chat_response(user_query: str, debug=False):
    casual = {
        "hi": "Hello! How can I assist you today?",
        "hey": "Hello! How can I assist you today?",
        "hello": "Hi there! How can I help you?",
        "thanks": "You're welcome!",
        "thank you": "Glad to help!",
        "ok": "👍", "okay": "👍",
        "bye": "Goodbye! Take care!",
    }
    low = user_query.strip().lower()
    if low in casual:
        return casual[low]

    context_items = _semantic_search(user_query, debug=debug)

    if not context_items:
        return "I'm Sorry, I couldn't find an answer. Please try rephrasing your question."

    # Deduplicate answers
    seen, unique_ctx = set(), []
    for c in context_items:
        if c["answer"] not in seen:
            unique_ctx.append(c)
            seen.add(c["answer"])
            if len(unique_ctx) >= TOP_K:
                break

    context_facts = "\n".join([f"- {c['answer']}" for c in unique_ctx])

    prompt = f"""
You are a helpful and knowledgeable AI Pathology Assistant.

Use the facts listed below to help answer the user's question. Even if the facts do not directly answer it, respond with related information if available. Do not say "I couldn't find an answer" if there are any medically relevant facts. Do not repeat the question in your answer.

Facts:
{context_facts}

User Question:
{user_query}

Answer concisely using the facts above (1-2 sentences):
"""


    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        resp = model.generate_content(prompt)
        answer = resp.text.strip()

        if not answer or "couldn't find" in answer.lower():
            return "I'm Sorry, I couldn't find an answer. Please try rephrasing your question."
        return answer

    except Exception as e:
        print("Gemini API error:", e)
        return "⚠️ Failed to get AI response."

response = get_chat_response("Which staining pattern is associated with ADA-SCID?", debug=True)
print(response)
