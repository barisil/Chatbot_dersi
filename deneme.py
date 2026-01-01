import os
import re
import streamlit as st
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import json
from typing import Literal, Dict, Any
import pandas as pd

# ---------------------------
# Config
# ---------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=500,
)

st.set_page_config(page_title="Çocuk Okul Eğitimi Chatbot", layout="wide")
st.title("📊 CSV Tabanlı RAG Sistemi (Gemini)")

DATA_FOLDER = "data"
PERSIST_DIR = "./chroma_db"


# ---------------------------
# Helpers
# ---------------------------
def parse_tuik_pipe_rows(file_path: str, encoding="utf-8"):
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


def build_docs_for_one_csv(file_path: str):
    rows = parse_tuik_pipe_rows(file_path)
    if not rows:
        return []

    dataset_name = os.path.basename(file_path)
    years = [r["year"] for r in rows]
    min_year, max_year = min(years), max(years)

    metrics = sorted({r["metric"] for r in rows if r["metric"]})
    breakdowns = sorted({r["breakdown"] for r in rows if r["breakdown"]})

    docs = []

    # 1️⃣ Dataset açıklama doc
    desc = (
        f"{dataset_name} dosyası TÜİK kaynaklı eğitim istatistiklerini içerir. "
        f"Yıl aralığı {min_year}-{max_year}. "
        f"Metrik örnekleri: {', '.join(metrics[:3])}. "
        f"Kırılım örnekleri: {', '.join(breakdowns[:3])}."
    )

    docs.append(Document(
        page_content=desc,
        metadata={"type": "dataset_description", "dataset": dataset_name}
    ))

    # 2️⃣ Satır bazlı doc’lar
    for r in rows[:200]:
        docs.append(Document(
            page_content=(
                f"{r['geo'] or 'Türkiye'} için {r['year']} yılında "
                f"{r['metric']} ({r['breakdown']}) değeri {r['value']} olarak raporlanmıştır."
            ),
            metadata={
                "type": "data_point",
                "dataset": dataset_name,
                "year": r["year"]
            }
        ))

    return docs

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)
    
@st.cache_resource
def load_all_rows_as_df():
    all_rows = []
    for fn in os.listdir(DATA_FOLDER):
        if fn.endswith(".csv"):
            fp = os.path.join(DATA_FOLDER, fn)
            rows = parse_tuik_pipe_rows(fp)
            # dosya bilgisini ekle
            for r in rows:
                r["source_file"] = fn
            all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame(columns=["metric","breakdown","year","value","geo","source_file"])

    df = pd.DataFrame(all_rows)
    # normalize string
    df["metric"] = df["metric"].fillna("").astype(str)
    df["breakdown"] = df["breakdown"].fillna("").astype(str)
    df["geo"] = df["geo"].fillna("Türkiye-TR").astype(str)
    return df

@st.cache_resource
def prepare_vector_db():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    # Eğer index varsa tekrar üretme
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

    docs = []
    for fn in os.listdir(DATA_FOLDER):
        if fn.endswith(".csv"):
            fp = os.path.join(DATA_FOLDER, fn)
            docs.extend(build_docs_for_one_csv(fp))

    if not docs:
        raise ValueError("Hiç doküman üretilemedi.")

    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

def extract_years(q: str):
    yrs = re.findall(r"\b(19|20)\d{2}\b", q)
    # re.findall yukarıdaki regexte grupluyor; daha basit:
    yrs = re.findall(r"\b(?:19|20)\d{2}\b", q)
    yrs = [int(y) for y in yrs]
    return yrs

def contains(q, *words):
    q = q.lower()
    return any(w.lower() in q for w in words)

def extract_gender_filter(q: str):
    ql = q.lower()
    # senin breakdown textlerinde "Cinsiyeti:Erkek" gibi geçiyor
    if "erkek" in ql and "kadın" in ql:
        return ["erkek", "kadın"]
    if "erkek" in ql:
        return ["erkek"]
    if "kadın" in ql:
        return ["kadın"]
    return []

def compute_aggregate_total(df: pd.DataFrame, question: str) -> str:
    years = extract_years(question)
    if not years:
        return "Hangi yıl için hesap yapmamı istersin?"
    year = years[0]

    # “özel eğitim” gibi bir filtreyi dosya adına/metric’e göre yakalayabiliriz
    # Basit: question’daki anahtarları metric/breakdown içinde ara
    ql = question.lower()

    sub = df[df["year"] == year].copy()

    # özel eğitim kelimesi geçiyorsa, dosya adı veya metrik/breakdown içinde ara
    if "özel" in ql:
        sub = sub[sub["source_file"].str.contains("ozel", case=False, na=False) |
                  sub["metric"].str.contains("özel", case=False, na=False) |
                  sub["breakdown"].str.contains("özel", case=False, na=False)]

    # öğretmen kelimesi geçiyorsa
    if "öğretmen" in ql or "ogretmen" in ql:
        sub = sub[sub["source_file"].str.contains("ogretmen", case=False, na=False) |
                  sub["metric"].str.contains("öğretmen", case=False, na=False) |
                  sub["breakdown"].str.contains("öğretmen", case=False, na=False)]

    genders = extract_gender_filter(question)
    if genders:
        # breakdown içinde erkek/kadın geçen satırları al
        mask = False
        for g in genders:
            mask = mask | sub["breakdown"].str.contains(g, case=False, na=False)
        sub = sub[mask]

    if sub.empty:
        return "Bu istek için ilgili veriyi bulamadım. (Yıl/özel eğitim/öğretmen kırılımı eşleşmedi.)"

    total = float(sub["value"].sum())
    # İstersen kırılım bazında da göster:
    top_groups = (
        sub.groupby("breakdown")["value"].sum().sort_values(ascending=False).head(10)
    )
    lines = [f"{year} yılı için toplam değer: {int(total) if total.is_integer() else total}"]
    lines.append("Bileşenler (kırılım bazında):")
    for idx, val in top_groups.items():
        lines.append(f"- {idx}: {int(val) if float(val).is_integer() else val}")

    return "\n".join(lines)


def compute_change(df: pd.DataFrame, question: str) -> str:
    years = extract_years(question)
    if len(years) < 2:
        return "Değişimi hesaplamak için iki yıl belirtmelisin (örn: 2015 ve 2016)."
    y1, y2 = years[0], years[1]

    ql = question.lower()
    sub = df.copy()

    # sorudan metric’i yakalamak zor; basitçe “net okullaşma” vs anahtarları kullan
    if "net" in ql:
        sub = sub[sub["source_file"].str.contains("net", case=False, na=False) |
                  sub["metric"].str.contains("net", case=False, na=False)]
    if "brüt" in ql or "brut" in ql:
        sub = sub[sub["source_file"].str.contains("brut", case=False, na=False) |
                  sub["metric"].str.contains("brüt", case=False, na=False)]

    genders = extract_gender_filter(question)
    if genders:
        mask = False
        for g in genders:
            mask = mask | sub["breakdown"].str.contains(g, case=False, na=False)
        sub = sub[mask]

    # seviye (ilkokul/ortaöğretim vb.)
    levels = ["ilkokul", "ortaöğretim", "ilköğretim", "okul öncesi", "mesleki", "genel ortaöğretim"]
    for lv in levels:
        if lv in ql:
            sub = sub[sub["breakdown"].str.contains(lv, case=False, na=False)]

    a = sub[sub["year"] == y1]["value"].mean()
    b = sub[sub["year"] == y2]["value"].mean()

    if pd.isna(a) or pd.isna(b):
        return "İki yıl için karşılaştırılabilir veri bulamadım (kırılım/metric uyuşmadı)."

    diff = b - a
    return f"{y1} → {y2} değişimi: {a:.1f} → {b:.1f}. Fark: {diff:+.1f} puan."


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

st.caption(f"✅ Vector DB hazır. Klasör: {PERSIST_DIR}")

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
"""Aşağıdaki bağlamı kullanarak soruyu cevapla.
Eğer bağlamda cevap yoksa "Bilmiyorum" de.

Bağlam:
{context}

Soru:
{question}

Cevap:"""
)


context_runnable = retriever | RunnableLambda(format_docs)





# IMPORTANT: prompt {input} bekliyor -> chain'de input anahtarını geçiyoruz
rag_chain = (
    {"context": context_runnable, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

def debug_retrieve(q: str):
    return retriever.invoke(q)   # langchain core yeni API


df_all = load_all_rows_as_df()

if user_query := st.chat_input("Sorunu yaz..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor..."):

            docs = debug_retrieve(user_query)

            with st.expander("🔎 Retrieved context"):
                st.write("\n\n---\n\n".join(d.page_content for d in docs))
                st.write([d.metadata for d in docs])

            if len(docs) == 0:
                answer = "Bu soru için veri setimde ilgili bir bağlam bulamadım."
            else:
                answer = rag_chain.invoke(user_query)

            st.write(answer)
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )


        
        



