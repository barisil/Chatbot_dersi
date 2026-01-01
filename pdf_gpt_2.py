import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd
import time
from typing import List, Dict, Tuple
from pathlib import Path


# Environment ayarları
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ Lütfen .env dosyasına OPENAI_API_KEY ekleyin.")
    st.stop()

# Streamlit config
st.set_page_config(
    page_title="TÜİK İstatistik Chatbot", 
    page_icon="📊",
    layout="wide"
)

st.title("📊 Türkiye Gençlik, Aile ve Yaşlı İstatistikleri Chatbot")
st.caption("OpenAI GPT + RAGAS ile performans değerlendirmeli versiyon")

# Sabitler
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

# LLM ve Embeddings
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000,
    api_key=api_key
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

# Session state başlatma
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexts" not in st.session_state:
    st.session_state.contexts = {}

# ============================================
# YARDIMCI FONKSİYONLAR
# ============================================

def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Dosya adından kategori ve yıl bilgisini çıkarır.
    Örnek: 'genclik_14.pdf' -> {'kategori': 'genclik', 'yil': '2014'}
    """
    metadata = {"kategori": "bilinmiyor", "yil": "bilinmiyor"}
    
    # Kategoriyi belirle
    if "genclik" in filename.lower():
        metadata["kategori"] = "genclik"
    elif "yasli" in filename.lower():
        metadata["kategori"] = "yasli"
    elif "aile" in filename.lower():
        metadata["kategori"] = "aile"
    
    # Yılı çıkar (14, 15, ... 24 formatında)
    year_match = re.search(r'_(\d{2})\.pdf', filename)
    if year_match:
        year_short = year_match.group(1)
        year_full = f"20{year_short}"
        metadata["yil"] = year_full
    
    return metadata

def load_all_pdfs(data_folder: str) -> List[Document]:
    """
    data/ klasöründeki tüm PDF'leri yükler ve metadata ekler.
    """
    all_documents = []
    
    if not os.path.exists(data_folder):
        st.error(f"❌ {data_folder} klasörü bulunamadı!")
        return all_documents
    
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        st.warning(f"⚠️ {data_folder} klasöründe PDF dosyası bulunamadı!")
        return all_documents
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_file in enumerate(pdf_files):
        pdf_path = os.path.join(data_folder, pdf_file)
        status_text.text(f"Yükleniyor: {pdf_file}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Metadata ekle
            file_metadata = extract_metadata_from_filename(pdf_file)
            
            for doc in documents:
                doc.metadata.update({
                    "source": pdf_file,
                    "kategori": file_metadata["kategori"],
                    "yil": file_metadata["yil"]
                })
            
            all_documents.extend(documents)
            
        except Exception as e:
            st.warning(f"⚠️ {pdf_file} yüklenemedi: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(pdf_files))
    
    progress_bar.empty()
    status_text.empty()
    
    return all_documents

def extract_years(text: str):
    years = re.findall(r"\b(20\d{2})\b", text)
    # unique, order-preserving
    seen = set()
    out = []
    for y in years:
        if y not in seen:
            seen.add(y)
            out.append(y)
    return out

def retrieve_docs_smart(question: str) -> List[Document]:
    years = extract_years(question)

    # Adaptif k: karşılaştırma varsa k büyüt
    base_k = TOP_K
    k = 12 if len(years) >= 2 else base_k

    docs_all: List[Document] = []

    # Yıl varsa: yıl yıl filtreli çek (en büyük iyileştirme)
    if years and st.session_state.vector_store is not None:
        for y in years:
            yr_retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": k, "filter": {"yil": y}}
            )
            docs_all.extend(yr_retriever.invoke(question))

        # Ek: Eğer filtreli arama az döndürdüyse, fallback genel arama
        if len(docs_all) < min(4, k):
            fallback = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": k}
            )
            docs_all.extend(fallback.invoke(question))

    else:
        # Yıl yoksa normal arama
        docs_all = st.session_state.retriever.invoke(question)

    # Dedupe (aynı chunk tekrar gelmesin)
    seen = set()
    unique_docs = []
    for d in docs_all:
        key = (
            d.metadata.get("source"),
            d.metadata.get("page"),
            d.metadata.get("kategori"),
            d.metadata.get("yil"),
            d.page_content[:120],
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    return unique_docs


def create_vector_store(documents: List[Document]) -> Chroma:
    """
    Dokümanlardan vektör veritabanı oluşturur.
    """
    with st.spinner("📝 Dokümanlar parçalanıyor..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        st.info(f"✂️ Toplam {len(splits)} metin parçası oluşturuldu")
    
    with st.spinner("🔢 Embeddings hesaplanıyor..."):
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
    
    return vector_store

def format_docs(docs: List[Document]) -> str:
    """
    Retrieve edilen dokümanları formatlar.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        kategori = doc.metadata.get('kategori', 'bilinmiyor')
        yil = doc.metadata.get('yil', 'bilinmiyor')
        content = doc.page_content
        formatted.append(f"[Kaynak {i} - {kategori.upper()} {yil}]\n{content}\n")
    return "\n".join(formatted)

def guard_mismatch(question: str, docs: List[Document]) -> bool:
    """
    True dönerse: cevap üretme, 'bulunamadı' de.
    Basit ama etkili: soru 'nüfus oranı' isterken context sadece 'bağımlılık oranı' veriyorsa engelle.
    """
    q = question.lower()
    ctx = " ".join(d.page_content for d in docs).lower()

    # Örnek kavram çakışması 1
    if "yaşlı nüfus oran" in q and "yaşlı bağımlılık oran" in ctx and "yaşlı nüfus oran" not in ctx:
        return True

    # Örnek kavram çakışması 2 (beklenen yaşam süresi türleri)
    if "beklenen yaşam süresi" in q:
        if "doğuşta" in q and ("65 yaş" in ctx or "65 yaşında" in ctx) and "doğuşta" not in ctx:
            return True
        if ("65 yaş" in q or "65 yaşında" in q) and "doğuşta" in ctx and ("65 yaş" not in ctx and "65 yaşında" not in ctx):
            return True

    return False


# ============================================
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = str(BASE_DIR / "data")
PERSIST_DIR = str(BASE_DIR / "chroma_db")

def init_vector_store():
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    documents = load_all_pdfs(DATA_FOLDER)
    if not documents:
        st.error(f"Veri bulunamadı: {DATA_FOLDER}")
        return None

    return create_vector_store(documents)

if st.session_state.vector_store is None:
    st.session_state.vector_store = init_vector_store()

if st.session_state.vector_store is not None and st.session_state.retriever is None:
    st.session_state.retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )


# ============================================
# RAG CHAIN
# ============================================

def create_rag_chain_no_retriever():
    template = """Sen TÜİK istatistiklerini analiz eden bir uzmansın.
SADECE verilen bağlam (context) içindeki ifadeleri kullanarak cevap ver.

Bağlam:
{context}

Soru:
{question}

Kurallar (çok önemli):
- Yalnızca bağlamda açıkça geçen bilgileri kullan. Genel bilgi ekleme.
- Bağlamda "dünya", "Avrupa", "OECD" gibi ifadeler yoksa bu tür karşılaştırmalar yapma.
- Kavramları karıştırma:
  "yaşlı nüfus oranı" ≠ "yaşlı bağımlılık oranı"
  "beklenen yaşam süresi (doğuşta)" ≠ "65 yaşında beklenen yaşam süresi"
  Soru hangi göstergeyi istiyorsa yalnızca o göstergenin değerini ver.
- Eğer sorulan gösterge bağlamda yoksa: "Bu bilgi verilen dokümanlarda bulunmamaktadır." de.
- Birden fazla yıl isteniyorsa, önce 2-3 cümleyle özeti yaz, sonra kısa bir tabloyla karşılaştır.
- Yılları doğru eşleştir. Yanlış yıl verme.

Sadece cevabı yaz. Etiket/başlık/format şablonu kullanma.

"""

    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()


GROUND_TRUTH_MAP = {
    "2020 yılında genç nüfus oranı nedir?":
        "2020 yılında genç nüfus, toplam nüfusun %15,4'ünü oluşturdu.",

    "2023 yılında akraba evliliği oranı nedir?":
        "2023 yılında akraba evliliği yapanların oranı %8,2 oldu",

    "2014 yılında boşanan çift sayısı kaçtır?":
        "Boşanan çift sayısı 2014 yılında 130 bin 913 oldu",

    "2018 yılında gençlerde işsizlik oranı nedir?":
        "2018 yılında gençlerde işsizlik oranı %20,3 oldu",

    "2020 yılında ne eğitimde ne istihdamda olan gençlerin oranı nedir?":
        "2020 yılında ne eğitimde ne istihdamda olan gençlerin oranı %28,3 oldu",

    "2023 yılında internet kullanan gençlerin oranı nedir?":
        "2023 yılında internet kullanan gençlerin oranı %97,5 oldu",

    "2024 yılında yaşlı nüfus kaç kişidir?":
        "2024 yılında yaşlı nüfus 9 milyon 112 bin 298 kişi oldu"
}

def norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

GROUND_TRUTH_MAP_N = {norm_q(k): v for k, v in GROUND_TRUTH_MAP.items()}



# ============================================
# ANA ALAN
# ============================================

# Eğer sistem hazırsa chat göster
if st.session_state.retriever:
    
    # RAG chain oluştur
    rag_chain = create_rag_chain_no_retriever()
    
    st.subheader("💬 Sohbet")
    
    # Mesaj geçmişini göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Eğer kaynaklar varsa göster
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Kaynaklar"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Kaynak {i}:** {source['source']}")
                        st.markdown(f"*Kategori:* {source['kategori']} | *Yıl:* {source['yil']}")
                        st.text(source['content'][:200] + "...")
                        st.divider()
    
    # Kullanıcı girişi
    if prompt := st.chat_input("Sorunuzu sorun..."):
        # Kullanıcı mesajı
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Bot cevabı
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                # tek retrieval (akıllı)
                retrieved_docs = retrieve_docs_smart(prompt)

                if not retrieved_docs:
                    response = "Bu bilgi verilen dokümanlarda bulunmamaktadır."
                    contexts = []
                elif guard:= (guard_mismatch(prompt, retrieved_docs)):
                    response = "Bu bilgi verilen dokümanlarda bulunmamaktadır."
                    contexts = [doc.page_content for doc in retrieved_docs]
                else:
                    context_text = format_docs(retrieved_docs)
                    response = rag_chain.invoke({"context": context_text, "question": prompt})
                    contexts = [doc.page_content for doc in retrieved_docs]

                # Ground truth ekle
                gt = GROUND_TRUTH_MAP_N.get(norm_q(prompt), "")
                # RAGAS için sakla (modelin gördüğü context ile aynı!)
                st.session_state.contexts[prompt] = {
                    "question": prompt,
                    "answer": response,
                    "contexts": contexts,
                    "ground_truth": gt
                    

                }

                st.markdown(response)
                
                # Kaynakları göster
                sources = []
                for doc in retrieved_docs:
                    sources.append({
                        "source": doc.metadata.get('source', 'Bilinmiyor'),
                        "kategori": doc.metadata.get('kategori', '-'),
                        "yil": doc.metadata.get('yil', '-'),
                        "content": doc.page_content
                    })
                
                with st.expander("📚 Kaynaklar"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Kaynak {i}:** {source['source']}")
                        st.markdown(f"*Kategori:* {source['kategori']} | *Yıl:* {source['yil']}")
                        st.text(source['content'][:200] + "...")
                        st.divider()
        
        # Mesajı kaydet
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "sources": sources
        })

else:
    # Sistem hazır değilse bilgilendirme
    st.info("👈 Lütfen sol menüden PDF'leri işleyip veritabanını oluşturun.")
    
    st.markdown("### 📋 Örnek Sorular")
    st.markdown("""
    - 2020 yılında gençlerin işsizlik oranı nedir?
    - 2014 ile 2024 arasında aile yapısı nasıl değişti?
    - Yaşlı nüfus oranı yıllara göre nasıl bir trend gösteriyor?
    - En son yıl için gençlik istatistikleri nedir?
    - Hangi yıllarda evlilik oranı en yüksekti?
    """)
    
    st.markdown("### 📊 Sistem Özellikleri")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model", "GPT-4o Mini")
        st.metric("Embedding", "text-embedding-3-small")
    
    with col2:
        st.metric("Chunk Size", CHUNK_SIZE)
        st.metric("Chunk Overlap", CHUNK_OVERLAP)
    
    with col3:
        st.metric("Top-K", TOP_K)
        st.metric("Temperature", 0.1)

# ============================================
# ALT BİLGİ
# ============================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    if st.session_state.messages:
        if st.button("🧹 Sohbeti Temizle"):
            st.session_state.messages = []
            st.session_state.contexts = {}
            st.rerun()

with col2:
    if st.session_state.contexts:
        st.info(f"✅ {len(st.session_state.contexts)} soru RAGAS için hazır")

with col3:
    if st.button("📈 RAGAS Değerlendirmesine Git"):
        st.switch_page("pages/ragas_evaluation.py")



