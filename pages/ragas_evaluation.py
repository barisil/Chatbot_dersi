import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset

st.set_page_config(page_title="RAGAS Evaluation", page_icon="📈", layout="wide")
st.title("📈 RAGAS Değerlendirmesi")

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ OPENAI_API_KEY bulunamadı (.env kontrol et).")
    st.stop()

if "contexts" not in st.session_state or not st.session_state.contexts:
    st.warning("Henüz değerlendirecek soru yok. Ana sayfada soru sorup contexts biriktir.")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
emb = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# Session -> DataFrame
rows = []
for q, item in st.session_state.contexts.items():
    gt = (item.get("ground_truth", "") or "").strip()
    rows.append({
        "question": item.get("question", q),
        "answer": item.get("answer", ""),
        "contexts": item.get("contexts", []),
        # RAGAS (senin sürümün) için doğru isim:
        "reference": gt
    })

df = pd.DataFrame(rows)

st.subheader("📄 Değerlendirilecek kayıtlar")
st.dataframe(df[["question", "answer", "reference"]], use_container_width=True)

st.divider()

st.subheader("⚙️ Metrik Seçimi")
col1, col2 = st.columns(2)
with col1:
    use_faithfulness = st.checkbox("faithfulness", value=True)
    use_answer_relevancy = st.checkbox("answer_relevancy", value=True)
with col2:
    use_context_precision = st.checkbox("context_precision (reference gerektirir)", value=True)
    use_context_recall = st.checkbox("context_recall (reference gerektirir)", value=True)

metrics_all = []
if use_faithfulness:
    metrics_all.append(faithfulness)
if use_answer_relevancy:
    metrics_all.append(answer_relevancy)

metrics_ref = []
if use_context_precision:
    metrics_ref.append(context_precision)
if use_context_recall:
    metrics_ref.append(context_recall)

if not metrics_all and not metrics_ref:
    st.error("En az 1 metrik seçmelisin.")
    st.stop()

if st.button("🚀 RAGAS'ı Çalıştır", type="primary"):
    with st.spinner("RAGAS skorları hesaplanıyor..."):
        # 1) reference istemeyen metrikler -> tüm sorular
        if metrics_all:
            dataset_all = Dataset.from_pandas(df[["question", "answer", "contexts"]])
            result_all = evaluate(
                dataset=dataset_all,
                metrics=metrics_all,
                llm=llm,
                embeddings=emb
            )
            res_all_df = result_all.to_pandas()

            st.subheader("✅ Sonuçlar (Tüm sorular / reference gerektirmeyen)")
            st.write("Genel (ortalama) skorlar:")
            st.json(res_all_df.mean(numeric_only=True).to_dict())
            st.write("Satır bazlı skorlar:")
            st.dataframe(res_all_df, use_container_width=True)

        # 2) reference isteyen metrikler -> sadece reference dolu sorular
        if metrics_ref:
            df_ref = df[df["reference"].astype(str).str.strip() != ""].copy()
            if df_ref.empty:
                st.warning("Reference (ground truth) dolu soru yok. context_precision / context_recall hesaplanamadı.")
            else:
                dataset_ref = Dataset.from_pandas(df_ref[["question", "answer", "contexts", "reference"]])
                result_ref = evaluate(
                    dataset=dataset_ref,
                    metrics=metrics_ref,
                    llm=llm,
                    embeddings=emb
                )
                res_ref_df = result_ref.to_pandas()

                st.subheader("✅ Sonuçlar (Sadece reference olan sorular)")
                st.write("Genel (ortalama) skorlar:")
                st.json(res_ref_df.mean(numeric_only=True).to_dict())
                st.write("Satır bazlı skorlar:")
                st.dataframe(res_ref_df, use_container_width=True)

                csv_bytes = res_ref_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Reference metrikleri CSV indir",
                    data=csv_bytes,
                    file_name="ragas_results_reference_only.csv",
                    mime="text/csv",
                )
