import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# Base directory
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent.parent

# Ruta absoluta al archivo de log
log_file_path = os.path.join(ROOT_DIR, "log", "app.log")

#Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),  # log a un archivo
        logging.StreamHandler()          # log a consola
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Read variables from env.
urls = os.getenv("SCRAPE_URLS", "").split(",")
pdf_path = BASE_DIR / os.getenv("PDF_PATH", "AI Engineer.pdf")
index_path = ROOT_DIR / os.getenv("VECTOR_INDEX_PATH", "promtior_index")

# Initialize document list
all_docs = []

# Load from URLs
for url in urls:
    logger.info(f"Loading content from: {url}")
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        all_docs.extend(docs)
    except Exception as e:
        logger.error(f"Error loading {url}: {e}")

# Load from PDF
if pdf_path and os.path.exists(pdf_path):
    logger.info(f"Loading content from PDF: {pdf_path}")
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        pages = pdf_loader.load()
        # Only includes page 3 - About Us (índex 2)
        all_docs.append(pages[2])
        """
        #include all the pdf
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        """
    except Exception as e:
        logger.error(f"Error loading PDF '{pdf_path}': {e}")
else:
    logger.error("PDF wasn`t found or specified.")

# Split documents into fragments
logger.info("Splitting documents...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(all_docs)

# Creating embeddings and FAISS index
logger.info("Generating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save índex
vectorstore.save_local(index_path)
logger.info(f"Index saved in: {index_path}")
