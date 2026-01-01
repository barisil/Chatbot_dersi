import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import CSVLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# ---------------------------
# Config
# ---------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📊 CSV Tabanlı RAG Sistemi (Gemini)")

DATA_FOLDER = "data"
PERSIST_DIR = "./chroma_db"


# ---------------------------
# Helpers
# ---------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


@st.cache_resource
def prepare_vector_db():
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError("Hata: 'data' klasörü bulunamadı!")

    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("Hata: 'data' klasöründe CSV dosyası bulunamadı!")

    file_path = os.path.join(DATA_FOLDER, csv_files[0])

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


    # Persist varsa aç
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Yoksa oluştur
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    documents = loader.load()

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )


# ---------------------------
# App
# ---------------------------
if not api_key:
    st.warning("Lütfen .env dosyasına GOOGLE_API_KEY ekleyin.")
    st.stop()

try:
    vector_store = prepare_vector_db()
except Exception as e:
    st.error(str(e))
    st.stop()

# Debug (istersen kapat)
try:
    st.caption(f"✅ Vector DB hazır. Klasör: {PERSIST_DIR}")
except Exception:
    pass

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                             temperature=0.3,
                             max_tokens=500)

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentences maximum.\n\n"
    "{context}"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

context_runnable = retriever | RunnableLambda(format_docs)

rag_chain = (
    {"context": context_runnable, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input("Sorunu yaz..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Gemini yanıtlıyor..."):
            answer = rag_chain.invoke(user_query)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})


