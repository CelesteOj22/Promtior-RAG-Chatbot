# test_search.py

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Variables igual que en app.py
index_path = "promtior_index"  # o usa os.getenv si querés
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Cargar índice
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Pregunta de prueba
query = "¿Cuándo se fundó Promtior?"

# Buscar los 5 documentos más similares
docs = vectorstore.similarity_search(query, k=5)

print(f"Resultados para la consulta: '{query}'\n")
for i, doc in enumerate(docs, 1):
    print(f"Fragmento {i}:")
    print(doc.page_content)
    print("-" * 50)

