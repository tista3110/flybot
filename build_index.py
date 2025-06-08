import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Step 1: Load CSV
df = pd.read_csv("qa_data.csv", encoding="windows-1252")  # your real file
questions = df["Question"].tolist()
answers = df["Answer"].tolist()

# Optional: remove empty rows
questions, answers = zip(*[(q, a) for q, a in zip(questions, answers) if isinstance(q, str) and isinstance(a, str)])

# Step 2: Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Step 3: Generate embeddings
embeddings = model.encode(list(questions), show_progress_bar=True)

# Step 4: Create FAISS index
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Step 5: Save index
faiss.write_index(index, "faiss_index.idx")

# Step 6: Save metadata
qa_data = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
with open("qa_metadata.pkl", "wb") as f:
    pickle.dump(qa_data, f)

print("✅ Embeddings and index saved successfully.")
