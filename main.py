from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import Body

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI()

# CORS agar bisa diakses dari floating-agent.html
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa dipersempit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Tambahkan memory sementara untuk session chat
chat_history = []
SYSTEM_PROMPT = (
    "Kamu adalah agen AI yang bertugas membantu pengguna memahami dan membangun workflow "
    "di n8n. Kamu bisa menjelaskan node, membaca kesalahan, menyarankan solusi dan "
    "membantu menulis kode atau ekspresi di n8n. Jangan menjawab di luar konteks n8n atau pengembangan workflow."
)

@app.get("/")
def serve_html():
    return FileResponse("static/floating-agent.html")

class NodeData(BaseModel):
    content: str  # Misalnya JSON node atau error log
    model: str = "gpt-4o-mini"  # default

# Endpoint chat sederhana
class ChatRequest(BaseModel):
    message: str
    model: str = "openai/gpt-4o"  # default model

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_msg = req.message
    model = req.model if hasattr(req, 'model') and req.model else "openai/gpt-4o"
    chat_history.append({"role": "user", "content": user_msg})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history[-10:]

    # Pilih endpoint dan API key sesuai model
    if model.startswith("groq/"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model.replace("groq/", ""),
            "messages": messages
        }
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "your-domain.com",
            "X-Title": "n8n-agent",
        }
        payload = {
            "model": model,
            "messages": messages
        }

    async with httpx.AsyncClient() as client:
        res = await client.post(url, headers=headers, json=payload)
        data = res.json()
        print("[DEBUG] Response dari API:", data)  # Logging sederhana

    if "choices" in data and data["choices"]:
        reply = data["choices"][0]["message"]["content"]
        chat_history.append({"role": "assistant", "content": reply})
        return { "reply": reply }
    else:
        error_msg = data.get("error", "Tidak ada balasan dari API atau API key salah/habis kuota.")
        return { "error": f"Gagal mendapatkan balasan dari API: {error_msg}. Response: {data}" }

@app.post("/ask-ai")
async def ask_ai(data: NodeData):
    try:
        if "claude" in data.model:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "your-domain.com",
                "X-Title": "n8n-agent",
            }
            payload = {
                "model": "anthropic/claude-3-haiku",
                "messages": [{"role": "user", "content": data.content}],
            }

        elif "gpt" in data.model:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "HTTP-Referer": "your-domain.com",
                "X-Title": "n8n-agent",
            }
            payload = {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": data.content}],
            }

        else:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": "llama3-8b-8192",  # atau mixtral-8x7b
                "messages": [{"role": "user", "content": data.content}],
            }

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()

        reply = result["choices"][0]["message"]["content"]
        return {"reply": reply}

    except Exception as e:
        return {"error": str(e)}
