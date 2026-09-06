# AI Chatbot Asesor de Pensiones IMSS 🇲🇽

> **Generative AI-Enabled Assistant for Mexican Social Security & Pension Queries**

A Streamlit-based web application designed to query, analyze, and synthesize official Mexican social security and IMSS (*Instituto Mexicano del Seguro Social*) pension documents using an advanced Retrieval-Augmented Generation (RAG) architecture powered by OpenRouter and LangChain.

* 🔗 **Live Application:** [AI Chatbot Asesor de Pensiones IMSS](https://chatbot-asesor-pensiones-imss.streamlit.app/)
* 🔗 **Read the article on Medium:** [Generative AI-Enabled Assistant for Pensions — Chatbot Asesor de Pensiones del IMSS](https://medium.com/latinxinai/generative-ai-enabled-assistant-for-pensions-a24435db6b01?sharedUserId=chomchom216)
* 🔗 **Read the article in Actuarios Trabajando (Mexico):** [Generative AI-Enabled Assistant for Pensions](https://static1.squarespace.com/static/694987771d60827a9dbf41de/t/69c7c8fc5c0536077e0c6acf/1774700797001/Revista+Volumen+17.pdf)

---

## 📌 Key Features

* **Multi-Document RAG Architecture**: Loads, normalizes, and indexes PDF regulatory documents stored in the local directory (`./pdf_files_seguridad_social`).
* **High-Availability OpenRouter Model Fallback**: Features automatic sequential model rotation across zero-cost inference endpoints on OpenRouter to protect against `429 Rate Limit Exceeded` errors and service outages:
  * `minimax/minimax-m3:free`
  * `nvidia/nemotron-3-ultra-550b-a55b:free`
  * `cohere/north-mini-code:free`
  * `google/gemma-4-31b-it:free`
  * `openrouter/free` *(Managed Dynamic Router Fallback)*
* **Intelligent Document Chunking**: Utilizes LangChain's `RecursiveCharacterTextSplitter` with `tiktoken` encoding to construct optimized context windows and prevent token truncation.
* **Spanish NLP & Greeting Handler**: Built-in time-aware greeting detection, accent normalization, and rule-based preprocessing for colloquial user inputs.
* **Citation Transparency**: Expands source context blocks showing exact document chunks used to compile each query response.

---

## 📁 Directory Structure

```text
.
├── pdf_files_seguridad_social/     # Target directory for official IMSS PDF documents
├── app3.3_spanish.py               # Main Streamlit application entry point
├── ai-advisor-icon.svg             # Application logo & branding asset
├── requirements.txt                # Required Python dependencies
├── .env                            # Environment variables configuration
└── README.md                       # Project documentation