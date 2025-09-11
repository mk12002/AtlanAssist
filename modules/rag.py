import os
from langchain_community.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI

EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-1.5-flash")
INDEX_PATH = "faiss_index"

def build_index():
    """
    Builds the FAISS index by crawling sitemaps from both the main
    documentation and the developer hub, then combining them.
    """
    print("Starting index build from Atlan Docs and Developer Hub...")
    loader_docs = SitemapLoader(web_path="https://docs.atlan.com/sitemap.xml", filter_urls=["https://docs.atlan.com/"])
    loader_dev = SitemapLoader(web_path="https://developer.atlan.com/sitemap.xml", filter_urls=["https://developer.atlan.com/"])
    
    print("Loading documents from both sitemaps...")
    docs_from_main = loader_docs.load()
    docs_from_dev = loader_dev.load()
    
    all_docs = docs_from_main + docs_from_dev
    print(f"Loaded a total of {len(all_docs)} documents.")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} text chunks.")
    
    print("Generating embeddings and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    vector_store.save_local(INDEX_PATH)
    print("✅ Index build complete and saved.")

def get_rag_chain():
    """Initializes the components needed for the RAG pipeline."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Please run build_index.py first.")
    vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
    return vector_store, llm

def get_rag_response_stream(vector_store, llm, query: str, conversation_history: str = ""):
    """
    Performs RAG and streams the response using pre-loaded components.

    Args:
        vector_store: The pre-loaded FAISS vector store.
        llm: The pre-loaded language model.
        query: The current user question.
        conversation_history: Previous conversation context for better follow-up answers.

    Yields:
        dict: A dictionary containing either a piece of the content ('chunk')
              or the final list of sources ('sources').
    """
    docs = vector_store.similarity_search(query, k=7)
    
    if not docs:
        yield {"chunk": "I could not find any relevant documents in the knowledge base to answer your question. Please try rephrasing your query or check the official Atlan documentation."}
        yield {"sources": []}
        return

    context = "\n\n".join([d.page_content for d in docs])
    sources = sorted(list(set(d.metadata.get("source", "") for d in docs)))

    conversation_section = ""
    if conversation_history.strip():
        conversation_section = f"""
**Previous Conversation:**
{conversation_history}
---"""

    prompt_template = f"""
You are an expert Atlan customer support assistant. Your primary goal is to provide a clear, direct, and helpful answer to the user's question using *only* the provided context.

Follow these instructions precisely:
1.  Analyze the provided **Context** to find the most relevant information to answer the user's **Question**.
2.  Synthesize a complete and comprehensive answer. Combine information from different sources in the context if needed.
3.  **Do not talk about the context itself.** Frame your response as a direct answer to the user. For example, instead of saying "The provided document states...", say "You can achieve this by...".
4.  If the context does not contain a perfect answer, do not give up. Provide the most relevant guidance you can find within the context. For instance, if the user asks about troubleshooting a crawler and the context only describes the setup process, you could say: "While I don't have specific troubleshooting steps, ensuring the initial setup is correct is the first step. The key configuration requirements are..."
5. **Clarification:** If the user's question is vague or open-ended (e.g., "what else?", "tell me more"), do not guess the topic. Instead, ask a clarifying question based on the previous topic. For example, you could ask: "Are you interested in more details about reporting, or would you like to switch to a new topic?"
6.  **Never** make up information or use knowledge outside the provided context.
{conversation_section}
**Context:**
---
{context}
---

**Current Question:** {query}

**Helpful Answer:**
"""
    for chunk in llm.stream(prompt_template):
        yield {"chunk": chunk.content}

    yield {"sources": sources}
