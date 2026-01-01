import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import pandas as pd

# ---------------------------
# Config
# ---------------------------
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

if not openai_key:
    st.error("⚠️ Lütfen .env dosyasına OPENAI_API_KEY ekleyin.")
    st.stop()

llm = ChatOpenAI(
    model="gpt-4o-mini",  # Ekonomik ve hızlı
    temperature=0.3,
    max_tokens=1000,
)

st.set_page_config(page_title="Çocuk Okul Eğitimi Chatbot", layout="wide")
st.title("📚 Çocuk Eğitimi İstatistikleri Chatbot")
st.caption("TÜİK verilerine dayalı eğitim istatistikleri analiz sistemi (OpenAI)")

DATA_FOLDER = "data"
PERSIST_DIR = "./chroma_db_openai"


# ---------------------------
# Helpers
# ---------------------------
def parse_tuik_pipe_rows(file_path: str, encoding="utf-8"):
    """TÜİK pipe-delimited formatını parse eder"""
    rows = []
    current_metric = None
    current_breakdown = None
    current_geo = None

    def is_year(s: str) -> bool:
        return s.isdigit() and 1900 <= int(s) <= 2100

    def parse_value(s: str):
        s = str(s).strip().replace(",", ".")
        try:
            return float(s)
        except:
            return None

    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if "Sütunlar" in line or line.startswith("Satırlar"):
                    continue

                parts = [p.strip() for p in line.split("|")]
                while parts and parts[0] == "":
                    parts.pop(0)
                while parts and parts[-1] == "":
                    parts.pop()

                if not parts:
                    continue

                if len(parts) == 1 and re.match(r".+-[A-Z]{2}$", parts[0]):
                    current_geo = parts[0]
                    continue

                if len(parts) >= 4 and is_year(parts[-2]):
                    metric, breakdown, year, value = parts[-4], parts[-3], parts[-2], parts[-1]
                    if metric:
                        current_metric = metric
                    if breakdown:
                        current_breakdown = breakdown

                    rows.append({
                        "metric": current_metric,
                        "breakdown": current_breakdown,
                        "year": int(year),
                        "value": parse_value(value),
                        "geo": current_geo
                    })
                    continue

                if len(parts) == 3 and is_year(parts[1]):
                    breakdown, year, value = parts
                    if breakdown:
                        current_breakdown = breakdown
                    rows.append({
                        "metric": current_metric,
                        "breakdown": current_breakdown,
                        "year": int(year),
                        "value": parse_value(value),
                        "geo": current_geo
                    })
                    continue

                if len(parts) == 2 and is_year(parts[0]):
                    year, value = parts
                    rows.append({
                        "metric": current_metric,
                        "breakdown": current_breakdown,
                        "year": int(year),
                        "value": parse_value(value),
                        "geo": current_geo
                    })
                    continue

                if len(parts) == 2:
                    m, b = parts
                    if m:
                        current_metric = m
                    if b:
                        current_breakdown = b

        return [r for r in rows if r["year"] and r["value"] is not None]
    
    except Exception as e:
        st.warning(f"⚠️ {file_path} dosyası okunurken hata: {str(e)}")
        return []


def build_docs_for_one_csv(file_path: str):
    """Bir CSV dosyasından Document nesneleri oluşturur"""
    rows = parse_tuik_pipe_rows(file_path)
    if not rows:
        return []

    dataset_name = os.path.basename(file_path).replace(".csv", "")
    years = [r["year"] for r in rows]
    min_year, max_year = min(years), max(years)

    metrics = sorted({r["metric"] for r in rows if r["metric"]})[:5]
    breakdowns = sorted({r["breakdown"] for r in rows if r["breakdown"]})[:5]

    docs = []

    # Dataset özet
    desc = (
        f"Veri Seti: {dataset_name}\n"
        f"Kapsam: TÜİK çocuk eğitimi istatistikleri\n"
        f"Yıl Aralığı: {min_year}-{max_year}\n"
        f"Ana Metrikler: {', '.join(metrics)}\n"
        f"Kırılımlar: {', '.join(breakdowns)}"
    )

    docs.append(Document(
        page_content=desc,
        metadata={
            "type": "dataset_summary",
            "dataset": dataset_name,
            "year_range": f"{min_year}-{max_year}"
        }
    ))

    # Yıllık özetler
    df_temp = pd.DataFrame(rows)
    for year in sorted(df_temp["year"].unique())[-5:]:  # Son 5 yıl
        year_data = df_temp[df_temp["year"] == year]
        summary = (
            f"{year} Yılı Özeti - {dataset_name}:\n"
            f"Toplam veri noktası: {len(year_data)}\n"
            f"Ortalama değer: {year_data['value'].mean():.2f}\n"
            f"Min: {year_data['value'].min()}, Max: {year_data['value'].max()}"
        )
        docs.append(Document(
            page_content=summary,
            metadata={
                "type": "year_summary",
                "dataset": dataset_name,
                "year": year
            }
        ))

    # Detaylı veri noktaları
    for r in rows[:80]:
        geo_text = r['geo'] if r['geo'] else 'Türkiye'
        
        content = (
            f"{geo_text}, {r['year']}: {r['metric']} - {r['breakdown']} = {r['value']}"
        )
        
        docs.append(Document(
            page_content=content,
            metadata={
                "type": "data_point",
                "dataset": dataset_name,
                "year": r["year"],
                "metric": r["metric"],
                "geo": geo_text
            }
        ))

    return docs


def format_docs(docs):
    if not docs:
        return "İlgili veri bulunamadı."
    return "\n\n".join(d.page_content for d in docs)


@st.cache_resource
def load_all_rows_as_df():
    """Tüm CSV dosyalarını DataFrame olarak yükler"""
    all_rows = []
    
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ {DATA_FOLDER} klasörü bulunamadı!")
        return pd.DataFrame(columns=["metric","breakdown","year","value","geo","source_file"])
    
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    
    if not csv_files:
        st.warning(f"⚠️ {DATA_FOLDER} klasöründe CSV dosyası bulunamadı!")
        return pd.DataFrame(columns=["metric","breakdown","year","value","geo","source_file"])
    
    for fn in csv_files:
        fp = os.path.join(DATA_FOLDER, fn)
        rows = parse_tuik_pipe_rows(fp)
        for r in rows:
            r["source_file"] = fn
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(columns=["metric","breakdown","year","value","geo","source_file"])

    df = pd.DataFrame(all_rows)
    df["metric"] = df["metric"].fillna("").astype(str)
    df["breakdown"] = df["breakdown"].fillna("").astype(str)
    df["geo"] = df["geo"].fillna("Türkiye-TR").astype(str)
    
    return df


@st.cache_resource
def prepare_vector_db():
    """Vector database'i hazırlar - OpenAI embeddings"""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Ekonomik embedding modeli
    )

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        st.info("♻️ Mevcut vector DB yükleniyor...")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

    st.info("🔄 Vector DB oluşturuluyor...")
    docs = []
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    
    if not csv_files:
        raise ValueError(f"❌ {DATA_FOLDER} klasöründe CSV dosyası bulunamadı!")
    
    for fn in csv_files:
        fp = os.path.join(DATA_FOLDER, fn)
        file_docs = build_docs_for_one_csv(fp)
        docs.extend(file_docs)
        st.caption(f"✓ {fn}: {len(file_docs)} doküman")

    if not docs:
        raise ValueError("❌ Hiç doküman üretilemedi!")

    st.info(f"📊 {len(docs)} doküman indexleniyor...")
    
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )


# ---------------------------
# RAG Chain Setup
# ---------------------------
try:
    vector_store = prepare_vector_db()
    st.success(f"✅ Vector DB hazır ({PERSIST_DIR})")
except Exception as e:
    st.error(f"❌ Vector DB hatası: {str(e)}")
    st.stop()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

prompt = ChatPromptTemplate.from_template(
"""Sen çocuk eğitimi istatistikleri uzmanısın. TÜİK verilerini kullanarak soruları yanıtlıyorsun.

Bağlam:
{context}

Soru: {question}

Kurallar:
- Sadece verilen bağlamdaki bilgileri kullan
- Bağlamda bilgi yoksa "Bu konuda veri bulamadım" de
- Sayıları net belirt, kaynak göster
- Kısa ve öz yanıt ver

Cevap:"""
)

context_runnable = retriever | RunnableLambda(format_docs)

rag_chain = (
    {"context": context_runnable, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# ---------------------------
# Streamlit UI
# ---------------------------
df_all = load_all_rows_as_df()

# Sidebar
with st.sidebar:
    st.header("📊 Veri Seti Bilgileri")
    
    if not df_all.empty:
        st.metric("Toplam Kayıt", len(df_all))
        st.metric("Yıl Aralığı", f"{df_all['year'].min()}-{df_all['year'].max()}")
        st.metric("Farklı Metrik", df_all['metric'].nunique())
        st.metric("CSV Dosyası", df_all['source_file'].nunique())
        
        with st.expander("📁 Dosyalar"):
            for file in df_all['source_file'].unique():
                st.write(f"• {file}")
    
    st.divider()
    debug_mode = st.checkbox("🔧 Debug Modu", value=False)
    
    if st.button("🗑️ Vector DB Sıfırla"):
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            st.success("Silindi. Sayfayı yenileyin.")
            st.rerun()

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.info("💡 **Örnek sorular:**")
    cols = st.columns(3)
    examples = [
        "2020 yılında okullaşma oranı nedir?",
        "İlkokul ve ortaokul karşılaştırması",
        "Son 5 yılda öğretmen sayısı değişimi"
    ]
    for col, q in zip(cols, examples):
        if col.button(q, key=f"ex_{q}"):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_query := st.chat_input("Sorunuzu yazın..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor..."):
            try:
                docs = retriever.invoke(user_query)
                
                if debug_mode:
                    with st.expander("🔍 Retrieved Context"):
                        for i, doc in enumerate(docs, 1):
                            st.write(f"**Doc {i}:**")
                            st.write(doc.page_content)
                            st.json(doc.metadata)
                
                if len(docs) == 0:
                    answer = "Bu soru için veri bulamadım. Farklı bir soru deneyin."
                else:
                    answer = rag_chain.invoke(user_query)
                
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            
            except Exception as e:
                error_msg = f"❌ Hata: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

st.divider()
st.caption("🔒 TÜİK verilerine dayalı | Powered by OpenAI GPT-4o-mini")