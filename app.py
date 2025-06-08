from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import logging, json
from datetime import datetime

# ---- local helpers ----
from utils import get_chat_response           # already does semantic search inside

# ---- FastAPI setup ----
app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

logging.basicConfig(filename="query_logs.log", level=logging.INFO)

def log_entry(query: str, answer: str) -> None:
    blob = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "ai_answer": answer
    }
    logging.info(json.dumps(blob))

# ---------- routes ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/", response_class=HTMLResponse)
async def ask_form(request: Request, query: str = Form(...)):
    ai_answer = get_chat_response(query)
    log_entry(query, ai_answer)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "response": {"query": query, "ai_answer": ai_answer}}
    )

@app.post("/query/")
async def ask_json(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()
    if not query:
        return {"ai_answer": "No input received."}

    ai_answer = get_chat_response(query)
    log_entry(query, ai_answer)
    return {"ai_answer": ai_answer}


if __name__ == "__main__":
    response = get_chat_response("Which staining pattern is associated with ADA-SCID?", debug=True)
    print(response)
