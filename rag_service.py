from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    model="netease-youdao/bce-embedding-base_v1"
)

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    model="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.3
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

text_chunks = []
chunk_embeds = []

def load_pdf(file_path: str):
    global text_chunks, chunk_embeds
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    chunks = text_splitter.split_documents(docs)
    chunk_text_list = [c.page_content for c in chunks]
    embeds = embedding.embed_documents(chunk_text_list)
    text_chunks = chunk_text_list
    chunk_embeds = np.array(embeds)
    return len(chunk_text_list)

def query_rag(question: str, top_k=3):
    q_emb = np.array(embedding.embed_query(question))
    sims = chunk_embeds @ q_emb
    top_idx = np.argsort(sims)[::-1][:top_k]
    context = "\n".join([text_chunks[i] for i in top_idx])
    prompt = f"""Use the context blow to answer the question.
Context:
{context}
Question: {question}
If you cannot find answer from context, say you don't know.
"""
    res = llm.invoke(prompt)
    return res.content
