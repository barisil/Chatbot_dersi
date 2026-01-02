import os
import re
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================
# YAPILANDIRMA

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Lütfen .env dosyasına OPENAI_API_KEY ekleyin.")
    st.stop()

# Sabitler
BASE_DIR = Path(__file__).resolve().parent
DATA_FOLDER = str(BASE_DIR / "data")
PERSIST_DIR = str(BASE_DIR / "chroma_db")

#Chatbotun en önemli kısımlarından bir tanesi. İdeal değerler değişebiliyor ama 
#benim dokümanlardaki text ler için genelde 500-800 arası chunk size ve %20 overlap iyi sonuç veriyor.


CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
TOP_K = 4

# Streamlit yapılandırma
st.set_page_config(
    page_title="TÜİK İstatistik Chatbot",
    page_icon="📊",
    layout="wide"
)

# LLM ve Embeddings GPT için
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000, api_key=api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

# Session state başlatma 
# Normalde streamlit her etkileşimde sayfayı tekrar başlatır, bunu yapmaması için session_state kullanılır.
for key, default in [
    ("vector_store", None),
    ("retriever", None),
    ("messages", []),
    ("contexts", {})
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Ground truth ölçümlemesi yapabilmek için kullandığımız test verileri
GROUND_TRUTH = {
    "2020 yılında genç nüfus oranı nedir?": "2020 yılında genç nüfus, toplam nüfusun %15,4'ünü oluşturdu.",
    "2023 yılında akraba evliliği oranı nedir?": "2023 yılında akraba evliliği yapanların oranı %8,2 oldu",
    "2014 yılında boşanan çift sayısı kaçtır?": "Boşanan çift sayısı 2014 yılında 130 bin 913 oldu",
    "2018 yılında gençlerde işsizlik oranı nedir?": "2018 yılında gençlerde işsizlik oranı %20,3 oldu",
    "2020 yılında ne eğitimde ne istihdamda olan gençlerin oranı nedir?": "2020 yılında ne eğitimde ne istihdamda olan gençlerin oranı %28,3 oldu",
    "2023 yılında internet kullanan gençlerin oranı nedir?": "2023 yılında internet kullanan gençlerin oranı %97,5 oldu",
    "2024 yılında yaşlı nüfus kaç kişidir?": "2024 yılında yaşlı nüfus 9 milyon 112 bin 298 kişi oldu"
}

# Kullanıcı sorusuyla eşleştirebilmek için normalize ediyoruz.
GROUND_TRUTH_NORM = {
    re.sub(r"\s+", " ", k.strip().lower()): v 
    for k, v in GROUND_TRUTH.items()
}

# ============================================
# YARDIMCI FONKSİYONLAR

# retrieval ve RAG için gerekli bir adım. Dosyalardaki kategori ve yıl bilgilerini çıkarır. Sonrasında bu bilgileri chunklara atayacağız.
# Verileri TÜİK ten yıl bazlı çektiğim için bu sınıflandırma işe yarıyor.
def extract_metadata(filename: str) -> Dict[str, str]:
    """Dosya adından kategori ve yıl çıkar."""
    metadata = {"kategori": "bilinmiyor", "yil": "bilinmiyor"}
    
    filename_lower = filename.lower()
    if "genclik" in filename_lower:
        metadata["kategori"] = "genclik"
    elif "yasli" in filename_lower:
        metadata["kategori"] = "yasli"
    elif "aile" in filename_lower:
        metadata["kategori"] = "aile"
    
    year_match = re.search(r'_(\d{2})\.pdf', filename)
    if year_match:
        metadata["yil"] = f"20{year_match.group(1)}"
    
    return metadata

# Belirtilen klasördeki tüm PDF dosyalarını sayfa bazında yükler ve her sayfaya yıl/kategori metadata’sı ekler.
# Bu fonksiyon sayesinde her chunk, hangi yıl ve kategoriye ait olduğunu biliyor
def load_pdfs(data_folder: str) -> List[Document]:
    if not os.path.exists(data_folder):
        st.error(f"❌ {data_folder} klasörü bulunamadı!")
        return []
    
    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        st.warning(f"⚠️ {data_folder} klasöründe PDF bulunamadı!")
        return []
    
    all_documents = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_file in enumerate(pdf_files):
        status_text.text(f"Yükleniyor: {pdf_file}")
        
        try:
            loader = PyPDFLoader(os.path.join(data_folder, pdf_file))
            documents = loader.load()
            file_metadata = extract_metadata(pdf_file)
            
            for doc in documents:
                doc.metadata.update({
                    "source": pdf_file,
                    **file_metadata
                })
            
            all_documents.extend(documents)
        except Exception as e:
            st.warning(f"⚠️ {pdf_file} yüklenemedi: {str(e)}")
        
        progress_bar.progress((idx + 1) / len(pdf_files))
    
    progress_bar.empty()
    status_text.empty()
    
    return all_documents

# Yüklenen dokümanları parçalara ayırır, her parçanın embedding’ini üretir ve kalıcı bir vektör veritabanı oluşturur.
def create_vector_store(documents: List[Document]) -> Chroma:
    """Vektör veritabanı oluştur."""
    with st.spinner("📝 Dokümanlar parçalanıyor..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = splitter.split_documents(documents)
        st.info(f"✂️ Toplam {len(splits)} metin parçası oluşturuldu")
# Her chunk için vektör oluşturuyoruz. Vektör veritabanını diske kalıcı olarak yazıyoruz, böylece her çalıştırmada embedding üretmiyoruz.
    with st.spinner("🔢 Embeddings hesaplanıyor..."):
        return Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )

def extract_years(text: str) -> List[str]:
    """Metinden yıl çıkar."""
    years = re.findall(r"\b(20\d{2})\b", text)
    return list(dict.fromkeys(years))  # Sırayı koruyarak unique yap


# Bu fonksiyon öncesi karşılaştırma sorularında bir yıla öncelik veriyordu. Şimdi tüm yıllar için farklı aramalar yapıyor ve sonuçları birleştiriyor.
def retrieve_docs(question: str) -> List[Document]:
    """Akıllı döküman retrieval."""
    years = extract_years(question)
    # eğer karşılaştırma yapılacaksa daha fazla sonuç almamız için k'yı artırıyoruz
    k = 12 if len(years) >= 2 else TOP_K
    
    docs_all = []
    
    # Yıl bazlı filtreleme
    if years and st.session_state.vector_store:
        for year in years:
            retriever = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": k, "filter": {"yil": year}}
            )
            docs_all.extend(retriever.invoke(question))
        
        # az sonuç varsa genel aramaya dönüyoruz
        if len(docs_all) < min(4, k):
            fallback = st.session_state.vector_store.as_retriever(
                search_kwargs={"k": k}
            )
            docs_all.extend(fallback.invoke(question))
    else:
        docs_all = st.session_state.retriever.invoke(question)
    
    # Tekrarları kaldır, context temizliği.
    unique_docs = []
    seen = set()
    for doc in docs_all:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page"),
            doc.metadata.get("kategori"),
            doc.metadata.get("yil"),
            doc.page_content[:120]
        )
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    
    return unique_docs

# Soru ile getirilen bağlam arasında kavramsal uyumsuzluk varsa, cevap üretilmesini engeller. 
# Yaşanan sorunlardan sonra ekledim. Daha kapsamlı arama sonucunda case sayısı arttırılabilir.

def guard_mismatch(question: str, docs: List[Document]) -> bool:
    q = question.lower()
    ctx = " ".join(d.page_content for d in docs).lower()
    
    # Yaşlı nüfus oranı vs yaşlı bağımlılık oranı
    if "yaşlı nüfus oran" in q and "yaşlı bağımlılık oran" in ctx and "yaşlı nüfus oran" not in ctx:
        return True
    
    # Beklenen yaşam süresi türleri
    if "beklenen yaşam süresi" in q:
        if "doğuşta" in q and ("65 yaş" in ctx or "65 yaşında" in ctx) and "doğuşta" not in ctx:
            return True
        if ("65 yaş" in q or "65 yaşında" in q) and "doğuşta" in ctx and "65 yaş" not in ctx and "65 yaşında" not in ctx:
            return True
    
    return False

# Metadata’yı veritabanında sütun olarak tutuyoruz; format_docs ise bu sütunları raporda başlık olarak gösteriyor.
# Metadata tek kez üretiliyor; sistem boyunca taşınıyor; LLM’e sadece okunur biçimde sunuluyor.
# Retriever doğru yıl chunk’ını getirir ama LLM bu bilginin hangi yıla ait olduğunu anlamayabilir, farklı chunk’ları karıştırabilir.
def format_docs(docs: List[Document]) -> str:
    """Dökümanları formatla."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        kategori = doc.metadata.get('kategori', 'bilinmiyor').upper()
        yil = doc.metadata.get('yil', 'bilinmiyor')
        formatted.append(f"[Kaynak {i} - {kategori} {yil}]\n{doc.page_content}\n")
    return "\n".join(formatted)

# Diskte kayıtlı bir vektör veritabanı varsa onu yükler; yoksa PDF’lerden sıfırdan oluşturur.
def init_vector_store():
    """Vektör veritabanını başlat veya yükle."""
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    documents = load_pdfs(DATA_FOLDER)
    if not documents:
        st.error(f"Veri bulunamadı: {DATA_FOLDER}")
        return None
    
    return create_vector_store(documents)

def create_rag_chain():
    """RAG chain oluştur."""
    template = """Sen TÜİK istatistik uzmanısın. Soruyu DOĞRUDAN ve KISACA cevapla.

Bağlam:
{context}

Soru:
{question}

CEVAP KURALLARI:
1. Soruyu DOĞRUDAN cevapla - gereksiz açıklama yapma
2. SADECE sorulan bilgiyi ver - ek detay ekleme
3. Sayısal bilgi varsa: "2023 yılında oran %15,4'tür." formatında ver
4. Karşılaştırma isteniyorsa: Kısa tablo veya liste kullan
5. "Bağlama göre...", "Kaynaklara göre..." gibi girişler kullanma
6. Kavram uyumsuzluğu varsa: "Bu bilgi dokümanlarda bulunmamaktadır."
7. **KRİTİK**: "Hangi yıl", "en fazla", "en az" sorularında:
   - SADECE bağlamda verilen yılları karşılaştır
   - Bağlamda olmayan yıl veya veri ASLA ekleme
   - Tüm yılların verisi yoksa: "Mevcut verilere göre [yıl] yılında [değer], ancak tüm yılların verisi bulunmamaktadır."


YASAKLAR:
❌ "Tabii ki", "Elbette", "Maalesef" gibi dolgu kelimeler
❌ Bağlamda olmayan genel bilgiler
❌ "Dünya", "Avrupa", "OECD" karşılaştırmaları (bağlamda yoksa)
❌ Kavram karıştırma: "yaşlı nüfus oranı" ≠ "yaşlı bağımlılık oranı"

Örnek İyi Cevap:
Soru: "2020'de genç işsizlik oranı nedir?"
Cevap: "2020 yılında genç işsizlik oranı %25,9'dur."
Örnek İyi Cevap (Karşılaştırma):
Soru: "Hangi yılda genç işsizlik en yüksekti?"
Cevap: "Mevcut verilere göre 2018 yılında genç işsizlik oranı %20,3 ile en yüksek seviyededir."


CEVAP (sadece cevap, başka hiçbir şey yazma):"""
    prompt = ChatPromptTemplate.from_template(template)
    return prompt | llm | StrOutputParser()

# ============================================
# BAŞLATMA

# Retriever, vektör veritabanına bağımlı. Bu yüzden önce DB’nin varlığını garanti altına alıp, sonra retriever’ı başlatıyorum.
if st.session_state.vector_store is None:
    st.session_state.vector_store = init_vector_store()

if st.session_state.vector_store and st.session_state.retriever is None:
    st.session_state.retriever = st.session_state.vector_store.as_retriever(
        search_kwargs={"k": TOP_K}
    )

# ============================================
# ANA UYGULAMA

#Bu bölümde chatbotun ana akışı var. Eğer vektör veritabanı hazırsa sohbet açılıyor. 
#Her kullanıcı sorusu için önce akıllı retrieval yapılıyor, ardından kavram uyumsuzluğu kontrol ediliyor. 
#Güvenliyse cevap üretiliyor; değilse ‘bilgi yok’ deniyor. Aynı anda tüm soru cevap context verisi RAGAS değerlendirmesi için loglanıyor."""

st.title("📊 Türkiye Gençlik, Aile ve Yaşlı İstatistikleri Chatbot")
st.caption("OpenAI GPT + RAGAS ile performans değerlendirmeli versiyon")

if st.session_state.retriever:
    rag_chain = create_rag_chain()
    
    st.subheader("💬 Sohbet")
    
    # Mesaj geçmişi
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Kaynaklar"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Kaynak {i}:** {source['source']}")
                        st.markdown(f"*Kategori:* {source['kategori']} | *Yıl:* {source['yil']}")
                        st.text(source['content'][:200] + "...")
                        st.divider()
    
    # Kullanıcı girişi
    if prompt := st.chat_input("Sorunuzu sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Düşünüyorum..."):
                retrieved_docs = retrieve_docs(prompt)
                
                if not retrieved_docs:
                    response = "Bu bilgi verilen dokümanlarda bulunmamaktadır."
                    contexts = []
                elif guard_mismatch(prompt, retrieved_docs):
                    response = "Bu bilgi verilen dokümanlarda bulunmamaktadır."
                    contexts = [doc.page_content for doc in retrieved_docs]
                else:
                    context_text = format_docs(retrieved_docs)
                    response = rag_chain.invoke({"context": context_text, "question": prompt})
                    contexts = [doc.page_content for doc in retrieved_docs]
                
                # Ground truth ve context kaydet
                norm_question = re.sub(r"\s+", " ", prompt.strip().lower())
                st.session_state.contexts[prompt] = {
                    "question": prompt,
                    "answer": response,
                    "contexts": contexts,
                    "ground_truth": GROUND_TRUTH_NORM.get(norm_question, "")
                }
                
                st.markdown(response)
                
                # Kaynakları göster
                sources = [
                    {
                        "source": doc.metadata.get('source', 'Bilinmiyor'),
                        "kategori": doc.metadata.get('kategori', '-'),
                        "yil": doc.metadata.get('yil', '-'),
                        "content": doc.page_content
                    }
                    for doc in retrieved_docs
                ]
                
                with st.expander("📚 Kaynaklar"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Kaynak {i}:** {source['source']}")
                        st.markdown(f"*Kategori:* {source['kategori']} | *Yıl:* {source['yil']}")
                        st.text(source['content'][:200] + "...")
                        st.divider()
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": sources
        })

else:
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
        st.metric("Temperature", 0)

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