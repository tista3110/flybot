import re, pickle, faiss, pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------- helper: same normalization as runtime --------
def normalize(text: str) -> str:
    """lowercase + strip punctuation + extra spaces"""
    # First convert to lowercase
    text = text.lower()
    # Remove all punctuation except question marks
    text = re.sub(r'[^\w\s?]', '', text)
    return text.strip()

# -------- configuration --------
CSV_PATH   = "qa_data.csv"
INDEX_PATH = "faiss_index_bge.idx"
META_PATH  = "qa_metadata_bge.pkl"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# --------------------------------

print("⏳ Loading CSV …")
df = pd.read_csv(CSV_PATH, encoding="windows-1252")
questions_raw = df["Question"].astype(str).tolist()
answers       = df["Answer"].astype(str).tolist()

# Clean + normalize
questions, clean_answers = [], []
for q, a in zip(questions_raw, answers):
    qn = normalize(q)
    if qn and a.strip():
        questions.append(qn)
        clean_answers.append(a.strip())

print(f"✅ {len(questions)} valid Q&A pairs")

print("⏳ Encoding embeddings …")
model = SentenceTransformer(EMBED_MODEL)
embeddings = model.encode(
    questions,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True  # cosine ready
)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

print("💾 Saving FAISS index …")
faiss.write_index(index, INDEX_PATH)

# Store both original and normalized questions
qa_data = [{"question": q, "normalized_question": qn, "answer": a} 
           for q, qn, a in zip(questions_raw, questions, clean_answers)]
with open(META_PATH, "wb") as f:
    pickle.dump(qa_data, f)

print("✅ Finished: index & metadata saved.")