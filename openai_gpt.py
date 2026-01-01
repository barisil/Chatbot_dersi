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

import pandas as pd
import time
from typing import List, Dict, Tuple

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision
)
from datasets import Dataset

# ---------------------------
# Config
# ---------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ Lütfen .env dosyasına OPENAI_API_KEY ekleyin.")
    st.stop()

st.set_page_config(page_title="Çocuk Okul Eğitimi Chatbot", layout="wide")
st.title("📚 Çocuk Eğitimi İstatistikleri Chatbot")
st.caption("OpenAI GPT + RAGAS ile performans değerlendirmeli versiyon")

DATA_FOLDER = "data"
PERSIST_DIR = "./chroma_db"

# LLM ve Embeddings
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000,
    api_key=api_key
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

# Sidebar ayarlar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    debug_mode = st.checkbox("🔧 Debug Modu", value=False, help="Hata mesajlarını detaylı göster")
    
    st.divider()
    
    max_docs_per_file = st.slider(
        "Dosya başına maksimum doküman",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
        help="Daha az doküman = daha az embedding kota kullanımı"
    )
    
    batch_size = st.slider(
        "Batch boyutu (embedding)",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
        help="Küçük batch = daha yavaş ama güvenli"
    )
    
    sleep_time = st.slider(
        "Batch arası bekleme (saniye)",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
        help="Rate limit için bekleme süresi"
    )
    
    retriever_k = st.slider(
        "Retriever K (döküman sayısı)",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
        help="Her sorguda kaç doküman getirilecek"
    )


# ---------------------------
# Helpers
# ---------------------------
def parse_tuik_pipe_rows(file_path: str, encoding="utf-8") -> List[Dict]:
    """TÜİK pipe-delimited formatını parse eder"""
    rows = []
    current_metric = None
    current_breakdown = None
    current_geo = None

    def is_year(s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        return s.isdigit() and 1900 <= int(s) <= 2100

    def parse_value(s: str):
        if not s:
            return None
        s = str(s).strip().replace(",", ".")
        s = re.sub(r'[^\d.\-]', '', s)
        try:
            return float(s)
        except:
            return None

    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            for line_num, raw in enumerate(f, 1):
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

                    parsed_value = parse_value(value)
                    if parsed_value is not None:
                        rows.append({
                            "metric": current_metric,
                            "breakdown": current_breakdown,
                            "year": int(year),
                            "value": parsed_value,
                            "geo": current_geo or "Türkiye-TR",
                            "source_line": line_num
                        })
                    continue

                if len(parts) == 3 and is_year(parts[1]):
                    breakdown, year, value = parts
                    if breakdown:
                        current_breakdown = breakdown
                    
                    parsed_value = parse_value(value)
                    if parsed_value is not None:
                        rows.append({
                            "metric": current_metric,
                            "breakdown": current_breakdown,
                            "year": int(year),
                            "value": parsed_value,
                            "geo": current_geo or "Türkiye-TR",
                            "source_line": line_num
                        })
                    continue

                if len(parts) == 2 and is_year(parts[0]):
                    year, value = parts
                    parsed_value = parse_value(value)
                    if parsed_value is not None:
                        rows.append({
                            "metric": current_metric,
                            "breakdown": current_breakdown,
                            "year": int(year),
                            "value": parsed_value,
                            "geo": current_geo or "Türkiye-TR",
                            "source_line": line_num
                        })
                    continue

                if len(parts) == 2:
                    m, b = parts
                    if m:
                        current_metric = m
                    if b:
                        current_breakdown = b

        valid_rows = [r for r in rows if r["year"] and r["value"] is not None]
        return valid_rows
    
    except Exception as e:
        st.warning(f"⚠️ {file_path} dosyası okunurken hata: {str(e)}")
        return []


def build_docs_for_one_csv(file_path: str, max_docs=30) -> List[Document]:
    """KOTA OPTİMİZASYONU: Minimum doküman ile maksimum bilgi"""
    rows = parse_tuik_pipe_rows(file_path)
    if not rows:
        return []

    dataset_name = os.path.basename(file_path).replace(".csv", "")
    df = pd.DataFrame(rows)
    
    docs = []
    
    years = sorted(df["year"].unique())
    metrics = df["metric"].unique()
    
    summary = (
        f"📊 {dataset_name}\n"
        f"Yıllar: {years[0]}-{years[-1]}\n"
        f"Metrikler: {', '.join(str(m) for m in metrics[:5])}\n"
        f"Toplam veri: {len(df)} kayıt\n"
        f"Ortalama değer: {df['value'].mean():.2f}"
    )
    
    docs.append(Document(
        page_content=summary,
        metadata={"type": "dataset_summary", "dataset": dataset_name}
    ))
    
    yearly_summaries = []
    for year in years[-5:]:
        year_data = df[df["year"] == year]
        yearly_summaries.append(
            f"{year}: Ort={year_data['value'].mean():.1f}, "
            f"Min={year_data['value'].min():.1f}, "
            f"Max={year_data['value'].max():.1f}"
        )
    
    if yearly_summaries:
        docs.append(Document(
            page_content=f"{dataset_name} - Yıllık İstatistikler:\n" + "\n".join(yearly_summaries),
            metadata={"type": "yearly_stats", "dataset": dataset_name}
        ))
    
    metric_summaries = []
    for metric in metrics[:5]:
        metric_data = df[df["metric"] == metric]
        metric_summaries.append(
            f"{metric}: {len(metric_data)} kayıt, Ort={metric_data['value'].mean():.1f}"
        )
    
    if metric_summaries:
        docs.append(Document(
            page_content=f"{dataset_name} - Metrikler:\n" + "\n".join(metric_summaries),
            metadata={"type": "metric_stats", "dataset": dataset_name}
        ))
    
    important_rows = []
    recent_years = sorted(df["year"].unique())[-3:]
    important_rows.extend(df[df["year"].isin(recent_years)].to_dict('records'))
    important_rows.extend(df.nlargest(5, 'value').to_dict('records'))
    important_rows.extend(df.nsmallest(5, 'value').to_dict('records'))
    
    seen = set()
    unique_rows = []
    for row in important_rows:
        key = (row['year'], row['metric'], row['breakdown'])
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    
    for row in unique_rows[:max_docs-3]:
        geo = row.get('geo', 'Türkiye-TR')
        content = (
            f"{geo}, {row['year']}: "
            f"{row['metric']} ({row['breakdown']}) = {row['value']}"
        )
        
        docs.append(Document(
            page_content=content,
            metadata={
                "type": "data_point",
                "dataset": dataset_name,
                "year": row["year"],
                "value": row["value"],
                "metric": row["metric"],
                "breakdown": row["breakdown"]
            }
        ))
    
    return docs


def format_docs(docs):
    """Dökümanları formatla"""
    if not docs:
        return "İlgili veri bulunamadı."
    return "\n\n".join(d.page_content for d in docs)


@st.cache_resource
def load_all_rows_as_df():
    """Tüm CSV dosyalarını DataFrame olarak yükler"""
    all_rows = []
    
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ {DATA_FOLDER} klasörü bulunamadı!")
        return pd.DataFrame()
    
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    
    if not csv_files:
        st.warning(f"⚠️ {DATA_FOLDER} klasöründe CSV dosyası bulunamadı!")
        return pd.DataFrame()
    
    for fn in csv_files:
        fp = os.path.join(DATA_FOLDER, fn)
        rows = parse_tuik_pipe_rows(fp)
        for r in rows:
            r["source_file"] = fn
        all_rows.extend(rows)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["metric"] = df["metric"].fillna("").astype(str)
    df["breakdown"] = df["breakdown"].fillna("").astype(str)
    df["geo"] = df["geo"].fillna("Türkiye-TR").astype(str)
    
    return df


def prepare_vector_db(max_docs_per_file, batch_size, sleep_time):
    """Vector DB hazırla"""
    
    # CSV dosyalarını kontrol et
    if not os.path.exists(DATA_FOLDER):
        st.error(f"❌ '{DATA_FOLDER}' klasörü bulunamadı!")
        st.info(f"💡 Lütfen '{DATA_FOLDER}' klasörü oluşturun ve CSV dosyalarınızı içine koyun")
        return None
    
    csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]
    
    if not csv_files:
        st.error(f"❌ '{DATA_FOLDER}' klasöründe CSV dosyası bulunamadı!")
        st.info("💡 Lütfen TÜİK CSV dosyalarınızı 'data/' klasörüne ekleyin")
        return None
    
    # Mevcut DB'yi kontrol et
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            st.info("♻️ Mevcut vector DB yükleniyor...")
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )

            # Embedding harcamadan DB dolu mu kontrol et
            try:
                _count = vectorstore._collection.count()
                st.info(f"📦 DB doküman sayısı: {_count}")
                if _count == 0:
                    raise ValueError("DB boş görünüyor")
            except Exception:
                pass

            st.success("✅ Mevcut DB başarıyla yüklendi")
            return vectorstore

        except Exception as e:
            st.warning(f"⚠️ Mevcut DB uyumsuz: {str(e)}")
            st.info("🔄 Yeni DB oluşturuluyor...")
            import shutil
            try:
                shutil.rmtree(PERSIST_DIR)
            except Exception as rm_err:
                st.warning(f"DB silinirken hata: {rm_err}")

      

    st.warning("🔄 Vector DB oluşturuluyor...")
    st.info(f"⚙️ Ayarlar: {len(csv_files)} dosya, Dosya başına {max_docs_per_file} doc, Batch={batch_size}")
    
    all_docs = []
    
    # Dökümanları topla
    for fn in csv_files:
        fp = os.path.join(DATA_FOLDER, fn)
        try:
            file_docs = build_docs_for_one_csv(fp, max_docs=max_docs_per_file)
            all_docs.extend(file_docs)
            st.caption(f"✓ {fn}: {len(file_docs)} doküman")
        except Exception as e:
            st.warning(f"⚠️ {fn} işlenirken hata: {str(e)}")
            continue
    
    if not all_docs:
        st.error("❌ Hiç doküman üretilemedi!")
        st.info("💡 CSV dosyalarınızın formatını kontrol edin")
        return None
    
    st.info(f"📦 Toplam {len(all_docs)} doküman, {(len(all_docs)-1)//batch_size + 1} batch'te işlenecek")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    vectorstore = None
    successful_batches = 0
    
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i+batch_size]
        batch_num = i//batch_size + 1
        total_batches = (len(all_docs)-1)//batch_size + 1
        
        status_text.text(f"⏳ Batch {batch_num}/{total_batches} işleniyor...")
        
        try:
            if vectorstore is None:
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=PERSIST_DIR
                )
            else:
                vectorstore.add_documents(batch)
            
            successful_batches += 1
            progress_bar.progress(min((i + batch_size) / len(all_docs), 1.0))
            
            if i + batch_size < len(all_docs) and sleep_time > 0:
                status_text.text(f"✓ Batch {batch_num} tamamlandı. {sleep_time}s bekleniyor...")
                time.sleep(sleep_time)
        
        except Exception as e:
            st.error(f"❌ Batch {batch_num} hatası: {str(e)}")
            
            # Rate limit kontrolü
            if "rate" in str(e).lower() or "quota" in str(e).lower() or "429" in str(e):
                st.warning("⚠️ Rate limit! 30 saniye bekleniyor...")
                time.sleep(30)
            else:
                st.warning("⏸️ 10 saniye bekleniyor...")
                time.sleep(10)
            
            # Tekrar dene
            try:
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=PERSIST_DIR
                    )
                else:
                    vectorstore.add_documents(batch)
                successful_batches += 1
            except Exception as e2:
                st.error(f"❌ Batch {batch_num} ikinci denemede de başarısız: {str(e2)}")
                # İlk batch başarısız olduysa dur
                if batch_num == 1:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ İlk batch oluşturulamadı. API key'inizi ve internet bağlantınızı kontrol edin.")
                    return None
                continue
    
    progress_bar.empty()
    status_text.empty()
    
    if vectorstore is None:
        st.error("❌ Vector DB oluşturulamadı!")
        return None
    
    st.success(f"✅ Vector DB oluşturuldu! ({successful_batches}/{total_batches} batch başarılı)")
    
    return vectorstore


def extract_years(query: str) -> List[int]:
    """Sorgudan yıl bilgilerini çıkar"""
    q = query.replace("–", "-").replace("—", "-")
    years = sorted(set(int(y) for y in re.findall(r"(19\d{2}|20\d{2})", q)))
    
    m = re.search(r"(19\d{2}|20\d{2})\s*-\s*(19\d{2}|20\d{2})", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a > b:
            a, b = b, a
        return list(range(a, b + 1))
    return years


def normalize_tr(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("İ", "i").replace("I", "ı")
    return s.lower()



def df_lookup_answer(df_all: pd.DataFrame, user_query: str, max_rows=50) -> Tuple[str, bool]:
    """Direkt DataFrame lookup - embedding kullanmadan"""
    if df_all is None or df_all.empty:
        return ("", False)

    q = normalize_tr(user_query)
    years = extract_years(q)

    keywords = [
    "eğitim","ortaöğretim","ilkokul","ortaokul","anaokulu","okul öncesi","kreş",
    "brüt","okullaşma","cinsiyet","oran","yıllık",
    "derslik","öğretmen","memnuniyet","devlet okulu","özel okul"
]    
    structured = (len(years) > 0) and any(k in q for k in keywords)

    if not structured:
        return ("", False)

    df = df_all.copy()

    if years:
        df = df[df["year"].isin(years)]

    # metrik hedefleme
    if "memnuniyet" in q:
        df = df[df["metric"].str.contains("memnuniyet", case=False, na=False)]

    if "derslik" in q:
        df = df[df["metric"].str.contains("Derslik", case=False, na=False)]

    if "öğretmen" in q:
        df = df[df["metric"].str.contains("Öğretmen", case=False, na=False)]

    if "kreş" in q or "anaokulu" in q or "okul öncesi" in q:
        df = df[df["metric"].str.contains("kreş|anaokul|okul önces", case=False, na=False)]

    if "ortaöğretim" in q:
        df = df[df["breakdown"].str.contains("Ortaöğretim", case=False, na=False)]
    if "ilkokul" in q:
        df = df[df["breakdown"].str.contains("İlkokul", case=False, na=False)]
    if "brüt" in q or "okullaşma" in q:
        df = df[df["metric"].str.contains("Okullaşma", case=False, na=False)]
    if "cinsiyet" in q:
        df = df[df["metric"].str.contains("Cinsiyet", case=False, na=False)]

    if "devlet" in q or "resmi" in q:
        df = df[df["breakdown"].str.contains("devlet|resmi", case=False, na=False)]

    if "özel" in q:
        df = df[df["breakdown"].str.contains("özel", case=False, na=False)]


    if df.empty:
        return ("", False)

    df = df.sort_values(["year", "metric", "breakdown"]).head(max_rows)

    lines = []
    for _, r in df.iterrows():
        geo = r.get("geo", "Türkiye-TR")
        lines.append(f"{r['year']}: {geo} | {r['metric']} ({r['breakdown']}) = {r['value']}")

    answer = "📊 Veritabanında bulunan eşleşen kayıtlar:\n\n" + "\n".join(lines)

    if years and (len(years) >= 2):
        found_years = set(df["year"].unique())
        missing = [y for y in years if y not in found_years]
        if missing:
            answer += f"\n\n⚠️ Not: Bu filtrelerle şu yıllar için kayıt çıkmadı: {missing}"

    return (answer, True)


# ---------------------------
# RAGAS Evaluation
# ---------------------------

def create_evaluation_dataset() -> List[Dict]:
    """Test için örnek veri seti oluştur"""
    return [
        {
            "question": "2020 yılında ilkokul okullaşma oranı nedir?",
            "ground_truth": "2020 yılı ilkokul okullaşma oranı hakkında bilgi"
        },
        {
            "question": "Ortaöğretim ve ilkokul öğrenci sayılarını karşılaştır",
            "ground_truth": "Ortaöğretim ve ilkokul öğrenci sayıları karşılaştırması"
        },
        {
            "question": "Son 5 yılda öğretmen sayısı nasıl değişti?",
            "ground_truth": "Son 5 yılda öğretmen sayısındaki değişim trendi"
        }
    ]


def evaluate_rag_with_ragas(rag_chain, retriever, test_questions: List[Dict]) -> Dict:
    """RAGAS ile RAG sistemini değerlendir"""
    
    st.info("🔄 RAGAS değerlendirmesi başlatılıyor...")
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    progress_bar = st.progress(0)
    
    for idx, item in enumerate(test_questions):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        try:
            answer = rag_chain.invoke(question)
            retrieved_docs = retriever.invoke(question)
            context = [doc.page_content for doc in retrieved_docs]
            
            questions.append(question)
            answers.append(answer)
            contexts.append(context)
            ground_truths.append(ground_truth)
            
            progress_bar.progress((idx + 1) / len(test_questions))
            
        except Exception as e:
            st.warning(f"⚠️ Soru işlenirken hata: {question[:50]}... | {str(e)}")
            continue
    
    progress_bar.empty()
    
    if not questions:
        st.error("❌ Hiç soru işlenemedi!")
        return {}
    
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    st.info("📊 RAGAS metrikleri hesaplanıyor...")

    try:
        result = evaluate(
            eval_dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_recall,
                context_precision,
            ]
        )

        # 🔹 Tabloya çevirmeyi dene (ragas >= 0.1.7)
        try:
            df_metrics = result.to_pandas()
            st.subheader("📈 RAGAS Sonuçları (Tablo)")
            st.dataframe(df_metrics)
        except Exception:
            st.subheader("📈 RAGAS Ham Sonuç")
            st.write(result)

        # 🔹 Tekil skorları güvenli şekilde çıkar
        scores = {}
        for metric in ["answer_relevancy", "faithfulness", "context_recall", "context_precision"]:
            try:
                scores[metric] = float(result[metric])
            except Exception:
                pass

        return scores

    except Exception as e:
        st.error(f"❌ RAGAS değerlendirme hatası: {str(e)}")
        return {}



# ---------------------------
# Main App
# ---------------------------

try:
    vector_store = prepare_vector_db(max_docs_per_file, batch_size, sleep_time)
    
    if vector_store is None:
        st.error("❌ Vector DB oluşturulamadı!")
        st.info("💡 Lütfen 'data/' klasöründe CSV dosyalarınızın olduğundan emin olun")
        st.stop()
    
    st.success("✅ Vector DB hazır")
    
except Exception as e:
    st.error(f"❌ Vector DB hatası: {str(e)}")
    st.info("💡 Olası çözümler:")
    st.info("1. 'data/' klasöründe CSV dosyaları var mı kontrol edin")
    st.info("2. OpenAI API key'inizin geçerli olduğunu kontrol edin")
    st.info("3. Sidebar'dan 'Vector DB Sıfırla' butonuna tıklayın")
    st.info("4. İnternet bağlantınızı kontrol edin")
    
    if debug_mode:
        st.exception(e)
    
    st.stop()

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": retriever_k,
        "filter": {"type": "data_point"}   # <-- kritik
    }
)

prompt = ChatPromptTemplate.from_template(
"""Sen çocuk eğitimi istatistikleri konusunda uzman bir asistansın. TÜİK verilerini kullanarak soruları yanıtlıyorsun.

📋 Bağlam (Retrieved Data):
{context}

❓ Soru: {question}

📌 Kurallar:
1. Sadece verilen bağlamdaki bilgileri kullan
2. Bağlamda bilgi yoksa "Bu konuda veri bulamadım" de ve alternatif öner
3. Sayıları ve istatistikleri net ve doğru belirt
4. Kısa, öz ve anlaşılır yanıt ver
5. Gerekirse karşılaştırma yap
6. Yıl bilgisini mutlaka belirt

✅ Cevap:"""
)

context_runnable = retriever | RunnableLambda(format_docs)

rag_chain = (
    {"context": context_runnable, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

df_all = load_all_rows_as_df()

# Sidebar stats & controls
with st.sidebar:
    st.divider()
    st.header("📊 İstatistikler")
    
    if not df_all.empty:
        st.metric("Toplam Kayıt", f"{len(df_all):,}")
        st.metric("Yıl Aralığı", f"{df_all['year'].min()}-{df_all['year'].max()}")
        st.metric("Dosya Sayısı", df_all['source_file'].nunique())
        st.metric("Benzersiz Metrik", df_all['metric'].nunique())
    else:
        st.warning("Veri yüklenemedi")
    
    st.divider()
    st.header("🧪 RAGAS Değerlendirme")
    
    if st.button("▶️ RAGAS Testi Çalıştır", type="primary"):
        with st.spinner("Test çalışıyor..."):
            test_data = create_evaluation_dataset()
            results = evaluate_rag_with_ragas(rag_chain, retriever, test_data)
            
            if results:
                st.success("✅ RAGAS değerlendirmesi tamamlandı!")
                
                st.subheader("📈 Metrik Sonuçları")
                
                metrics_df = pd.DataFrame([results])
                st.dataframe(metrics_df)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'answer_relevancy' in results:
                        st.metric("Answer Relevancy", f"{results['answer_relevancy']:.3f}")
                    if 'faithfulness' in results:
                        st.metric("Faithfulness", f"{results['faithfulness']:.3f}")
                
                with col2:
                    if 'context_recall' in results:
                        st.metric("Context Recall", f"{results['context_recall']:.3f}")
                    if 'context_precision' in results:
                        st.metric("Context Precision", f"{results['context_precision']:.3f}")
    
    st.divider()
    
    if st.button("🗑️ Vector DB Sıfırla", type="secondary"):
        import shutil
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
            st.success("✅ Silindi! Sayfayı yenileyin.")
            st.rerun()

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

if len(st.session_state.messages) == 0:
    st.info("💡 **Örnek sorular:**")
    cols = st.columns(3)
    examples = [
        "2020 yılı okullaşma oranı nedir?",
        "İlkokul ve ortaokul karşılaştır",
        "Son yıllarda öğretmen sayısı"
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
                lookup_text, matched = df_lookup_answer(df_all, user_query)

                if matched:
                    answer = lookup_text
                else:
                    docs = retriever.invoke(user_query)

                    if debug_mode:
                        with st.expander("🔍 Retrieved Contexts"):
                            for i, doc in enumerate(docs, 1):
                                st.write(f"**Doc {i}:**")
                                st.code(doc.page_content)
                                st.json(doc.metadata)

                    if not docs:
                        answer = "❌ Bu soru için veri bulamadım. Lütfen sorunuzu farklı şekilde ifade edin."
                    else:
                        answer = rag_chain.invoke(user_query)

                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"❌ Hata: {str(e)}")
                if debug_mode:
                    st.exception(e)

st.divider()
st.caption("🔒 TÜİK verileri | OpenAI: gpt-4o-mini + text-embedding-3-small")

