import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="DDC AI Classifier",
    layout="centered", # Mengubah layout menjadi centered agar lebih fokus
    initial_sidebar_state="collapsed"
)

# 2. FUNGSI LOAD MODEL
# Pastikan folder model sudah sesuai dengan path ini
MODEL_PATH = "./model_distilbert_final"

@st.cache_resource
def load_model_resources():
    if not os.path.exists(MODEL_PATH):
        return None, None, "Model path not found"
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)

# 3. HEADER
st.title("DDC AI Classifier")
st.write("Automated Dewey Decimal Classification using DistilBERT")
st.divider()

# 4. INIT MODEL
tokenizer, model, error_msg = load_model_resources()

if error_msg:
    st.error(f"System Error: Failed to load model. {error_msg}")
    st.stop()

# 5. DATA LABEL DDC
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

DDC_DESCRIPTIONS = {
    0: "Computer science, information systems, encyclopedias, and general reference works",
    1: "Philosophical theories, psychology, ethics, and human behavior studies",
    2: "Religious texts, theology, comparative religion, and spiritual practices",
    3: "Sociology, economics, political science, law, education, and social issues",
    4: "Linguistics, grammar, dictionaries, and language learning materials",
    5: "Mathematics, astronomy, physics, chemistry, biology, and earth sciences",
    6: "Medicine, engineering, agriculture, manufacturing, and applied sciences",
    7: "Fine arts, music, performing arts, sports, games, and recreational activities",
    8: "Poetry, novels, literary criticism, and works of fiction and prose",
    9: "World history, biography, geography, and area studies"
}

# 6. TABS NAVIGASI
tab1, tab2 = st.tabs(["Classification", "About DDC"])

# ==================== TAB 1: CLASSIFICATION ====================
with tab1:
    st.write("Input the book details below to classify.")
    
    with st.form("classification_form"):
        title_input = st.text_input("Book Title")
        desc_input = st.text_area("Description / Abstract", height=150)
        
        submit_btn = st.form_submit_button("Classify Text", type="primary")

    if submit_btn:
        if not title_input or not desc_input:
            st.warning("Please provide both book title and description.")
        else:
            # Proses Progress
            progress_bar = st.progress(0, text="Processing...")
            for percent_complete in range(100):
                time.sleep(0.005)
                progress_bar.progress(percent_complete + 1)
            
            # Inferensi
            try:
                text_combined = f"{title_input} {title_input} {desc_input}"
                inputs = tokenizer(text_combined, return_tensors="pt", truncation=True, max_length=256)
                
                model.eval()
                with torch.no_grad():
                    outputs = model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].numpy()
                pred_idx = np.argmax(probs)
                confidence_score = probs[pred_idx]
                
                progress_bar.empty()

                # Tampilan Hasil
                st.subheader("Result")
                
                # Container untuk hasil utama
                with st.container(border=True):
                    st.metric(
                        label="Predicted Class", 
                        value=DDC_LABELS[pred_idx]
                    )
                    st.write(DDC_DESCRIPTIONS[pred_idx])
                    
                    st.markdown("---")
                    st.write(f"Confidence Score: **{confidence_score:.1%}**")
                    
                    if confidence_score > 0.8:
                        st.success("High Confidence")
                    elif confidence_score > 0.6:
                        st.info("Good Confidence")
                    else:
                        st.warning("Low Confidence - Check input")

                # Grafik Probabilitas
                st.subheader("Probability Distribution")
                
                top5_indices = np.argsort(probs)[-5:][::-1]
                chart_data = pd.DataFrame({
                    "Class": [f"{i}00" for i in top5_indices],
                    "Probability": [probs[i] for i in top5_indices]
                })
                
                st.bar_chart(chart_data.set_index("Class"))
                
                # Tabel Data
                with st.expander("View Detailed Statistics"):
                    st.table(chart_data)

            except Exception as e:
                st.error(f"Error during classification: {str(e)}")

    # Daftar Kategori (Disederhanakan dalam Expander)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Reference: DDC Main Classes"):
        for code, text in DDC_LABELS.items():
            st.text(f"{text}")

# ==================== TAB 2: ABOUT ====================
with tab2:
    st.subheader("Dewey Decimal Classification")
    st.write("""
    The Dewey Decimal Classification (DDC) system organizes knowledge into ten main classes. 
    This system uses a DistilBERT model trained on library records to predict the correct category based on book titles and descriptions.
    """)
    
    st.subheader("The Ten Main Classes")
    
    # Menggunakan dataframe sederhana untuk menampilkan list agar rapi
    data_classes = {
        "Code": ["000", "100", "200", "300", "400", "500", "600", "700", "800", "900"],
        "Category": [
            "Computer science, information & general works",
            "Philosophy & psychology",
            "Religion",
            "Social sciences",
            "Language",
            "Science",
            "Technology",
            "Arts & recreation",
            "Literature",
            "History & geography"
        ]
    }
    st.table(pd.DataFrame(data_classes))
    
    st.subheader("Model Information")
    st.write("""
    - **Model Architecture:** DistilBERT
    - **Input:** Title + Description
    - **Output:** DDC Main Class (000-900)
    """)

# FOOTER SIMPLE
st.markdown("---")
st.caption("DDC AI Classifier System")