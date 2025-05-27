import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from uuid import uuid4

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langserve import add_routes


from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_together import Together


# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Ruta absoluta al archivo de log
log_file_path = os.path.join(BASE_DIR, "log", "app.log")

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # log a un archivo
        logging.StreamHandler()              # log a consola
    ]
)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()

# Read variables from .env
index_path = os.getenv("VECTOR_INDEX_PATH", "promtior_index")

# Templates and FastAPI
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app = FastAPI()

# Load FAISS index
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Index found, loading...")
    vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vs

# Instantiate vectorstore, LLM and RAG chain
vectorstore = get_vectorstore()
llm = Together(
    model="meta-llama/Llama-3-8b-chat-hf",
    temperature=0.7,
    max_tokens=512,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
# Add RAG route
add_routes(app, rag_chain, path="/chat")

# Chat Histories per session
chat_histories = {}


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    # Generar un session_id único
    session_id = str(uuid4())
    # Inicializar historial para esta sesión
    chat_histories[session_id] = []

    return templates.TemplateResponse("form.html", {
        "request": request,
        "history": [],
        "session_id": session_id
    })


@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, question: str = Form(...), session_id: str = Form(...)):
    logger.info(f"Received question: {question}")
    # Obtener o inicializar historial de la sesión
    history = chat_histories.get(session_id, [])

    # Obtener respuesta del RAG
    output = rag_chain.invoke({"query": question})
    answer = output.get("result")
    logger.info(f"Generated answer: {answer}")

    # Guardar en historial
    history.append({"question": question, "answer": answer})
    chat_histories[session_id] = history

    return templates.TemplateResponse("form.html", {
        "request": request,
        "history": history,
        "session_id": session_id
    })

