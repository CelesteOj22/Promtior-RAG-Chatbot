# test_search.py

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Variables
index_path = "promtior_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# load index
vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

# Test Query
query = "¿When whas Promtior founded?"

# Search the 5 most relevant documents
docs = vectorstore.similarity_search(query, k=5)

print(f"Query Results: '{query}'\n")
for i, doc in enumerate(docs, 1):
    print(f"Fragment {i}:")
    print(doc.page_content)
    print("-" * 50)

