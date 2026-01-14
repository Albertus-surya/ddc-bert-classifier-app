import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="DDC Classifier System",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk tampilan lebih bersih (Menghapus padding berlebih)
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #0e1117;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# 2. FUNGSI LOAD MODEL
MODEL_PATH = "./model_distilbert_final"

@st.cache_resource
def load_model_resources():
    # Cek keberadaan folder model
    if not os.path.exists(MODEL_PATH):
        return None, None, "Model path not found"
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


# 3. HEADER & JUDUL
st.title("Automated Library Classification")
st.markdown("""
<div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 5px solid #2196F3; margin-bottom: 25px;'>
    <small style='color: #666;'>SYSTEM DESCRIPTION</small><br>
    This system utilizes <b>DistilBERT architecture</b> to predict the Dewey Decimal Classification (DDC) 
    based on English book titles and descriptions.
</div>
""", unsafe_allow_html=True)


# 4. LOAD MODEL (Background)
with st.spinner("Initializing neural network model..."):
    tokenizer, model, error_msg = load_model_resources()

if error_msg:
    st.error(f"System Error: Failed to load model resources. {error_msg}")
    st.stop()

# CONSTANTS
DDC_LABELS = {
    0: "000 - Generalities & Computer Science",
    1: "100 - Philosophy & Psychology",
    2: "200 - Religion",
    3: "300 - Social Sciences",
    4: "400 - Language",
    5: "500 - Natural Sciences",
    6: "600 - Technology (Applied Sciences)",
    7: "700 - The Arts & Recreation",
    8: "800 - Literature & Rhetoric",
    9: "900 - History & Geography"
}

# INPUT FORM (Clean Interface)
with st.container(border=True):
    st.subheader("Input Book Data")
    
    with st.form("classification_form"):
        title_input = st.text_input("Book Title", placeholder="Enter the exact title of the book")
        desc_input = st.text_area("Description / Abstract", placeholder="Paste the book description here...", height=120)
        
        # Tombol Submit full width
        submit_btn = st.form_submit_button("Run Classification", type="primary", use_container_width=True)


#  LOGIKA PREDIKSI & HASIL
if submit_btn:
    if not title_input or not desc_input:
        st.warning("Please provide both Title and Description to proceed.")
    else:
        # Progress bar visual untuk UX
        progress_text = "Analyzing semantic patterns..."
        my_bar = st.progress(0, text=progress_text)

        try:
            # Simulasi progress (cepat)
            for percent_complete in range(0, 100, 20):
                time.sleep(0.05)
                my_bar.progress(percent_complete + 20, text=progress_text)

            # --- PROSES INFERENSI ---
            text_combined = f"{title_input} {title_input} {desc_input}"
            inputs = tokenizer(text_combined, return_tensors="pt", truncation=True, max_length=256)
            
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Kalkulasi Probabilitas
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()
            pred_idx = np.argmax(probs)
            confidence_score = probs[pred_idx]
            
            # Selesai progress
            my_bar.empty()

            # --- TAMPILAN HASIL (DASHBOARD STYLE) ---
            st.markdown("---")
            st.subheader("Classification Results")

            # Container Hasil Utama
            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.caption("PREDICTED DDC CLASS")
                    # Menampilkan hasil dengan font besar
                    st.markdown(f"<h2 style='margin-top: -10px; color: #1f77b4;'>{DDC_LABELS[pred_idx]}</h2>", unsafe_allow_html=True)
                    
                    # Analisis singkat
                    st.info(f"The model has identified patterns matching **Class {pred_idx}00** with high confidence.")

                with col2:
                    st.caption("CONFIDENCE SCORE")
                    # Metric style
                    st.metric(label="", value=f"{confidence_score:.1%}")
                    # Progress bar untuk confidence
                    st.progress(float(confidence_score))

            # --- TAMPILAN CHART (CLEAN) ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("Probability Distribution (Top 3)")
            
            # Data Processing untuk Chart
            top3_indices = np.argsort(probs)[-3:][::-1]
            
            chart_data = []
            for i in top3_indices:
                chart_data.append({
                    "Category": str(i*100), # Hanya ambil angka ratusan (000, 100) agar grafik rapi
                    "Probability": probs[i],
                    "Full Label": DDC_LABELS[i]
                })
            
            df_chart = pd.DataFrame(chart_data)

            # Menggunakan column config untuk tampilan tabel yang lebih pro
            col_chart, col_table = st.columns([2, 1])

            with col_chart:
                 st.bar_chart(
                    df_chart, 
                    x="Category", 
                    y="Probability",
                    color="#2196F3",
                    height=250
                )

            with col_table:
                st.caption("Detailed Breakdown")
                # Menampilkan tabel mini tanpa index
                st.table(df_chart[["Category", "Probability"]].assign(Probability=lambda x: x['Probability'].map('{:.2%}'.format)))

        except Exception as e:
            st.error(f"An error occurred during classification: {str(e)}")

# FOOTER
st.markdown("""
    <div style='text-align: center; margin-top: 50px; color: #888; font-size: 12px;'>
        &copy; 2025 AI Library Systems. Powered by DistilBERT Transformer Model.
    </div>
""", unsafe_allow_html=True)