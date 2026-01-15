# Automated Library Classification (DDC AI)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ddc-bert-classifier-app.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Library-HuggingFace_Transformers-yellow)
![Model](https://img.shields.io/badge/Model-DistilBERT-red)

An AI-powered application designed to automatically classify books into the **Dewey Decimal Classification (DDC)** system based on their English titles and descriptions.

This system utilizes a fine-tuned **DistilBERT** architecture, optimized for deployment with **Safetensors** serialization for security and speed.

---
## Key Features

* **Smart Classification:** Predicts one of the 10 main DDC classes (000-900).
* **High Performance:** Achieves **~80.68% accuracy** on the validation dataset.
* **Fast & Lightweight:** Uses DistilBERT.
* **Professional UI:** Built with Streamlit, featuring confidence visualization and probability breakdown.

## Demo
This application is deployed using Streamlit.  
**([Demo Streamlit](https://ddc-bert-classifier-app.streamlit.app/))**

## About Model
This model was built using the **DistilBERT** (Transformer) architecture that has been fine-tuned.
* **Base Model:** `distilbert-base-uncased`
* **Dataset:** Book data from OpenLibrary (Dataset cleaned and balanced).
* **Input:** Book Title & Description.
* **Output:** One of 10 Main DDC Classes (000-900).
* **Model Accuracy:** ~80% (on validation data).

## Model Architecture

The model is fine-tuned on a balanced dataset extracted from **OpenLibrary Data Dumps**.

* **Base Model:** `distilbert-base-uncased`
* **Input:** Title + Description (Concatenated)
* **Output:** Softmax probabilities for 10 classes.

### Supported DDC Classes:
| Code | Category |
| :--- | :--- |
| **000** | Computer Science, Information & General Works |
| **100** | Philosophy & Psychology |
| **200** | Religion |
| **300** | Social Sciences |
| **400** | Language |
| **500** | Science |
| **600** | Technology |
| **700** | Arts & Recreation |
| **800** | Literature |
| **900** | History & Geography |

---

##  Repository Structure

```text
ddc-classifier-app/
├── model_distilbert_final/   # Pre-trained Model Artifacts
│   ├── config.json           # Model configuration
│   ├── model.safetensors     # Model weights (stored via Git LFS)
│   ├── tokenizer.json        # Tokenizer data
│   └── vocab.txt             # Vocabulary
├── app.py                    # Main Streamlit Application
├── requirements.txt          # Python Dependencies
└── README.md                 # Project Documentation
