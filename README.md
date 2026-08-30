# AI Chatbot Asesor de Pensiones IMSS 🇲🇽

> **Generative AI-Enabled Assistant for Mexican Social Security & Pension Queries**

A Streamlit-based web application designed to query, analyze, and synthesize official Mexican social security and IMSS (Instituto Mexicano del Seguro Social) pension documents using advanced Retrieval-Augmented Generation (RAG) architectures powered by OpenRouter and LangChain.

🔗 **Live Application:** [AI Chatbot Asesor de Pensiones IMSS](https://chatbot-asesor-pensiones-imss.streamlit.app/) 
🔗 **Read the article at Medium:** [Generative AI-Enabled Assistant for Pensions Chatbot Asesor de Pensiones del IMSS](https://medium.com/latinxinai/generative-ai-enabled-assistant-for-pensions-a24435db6b01?sharedUserId=chomchom216)
🔗 **Read the article at Actuarios Trabajando - México:** [Generative AI-Enabled Assistant for Pensions](https://static1.squarespace.com/static/694987771d60827a9dbf41de/t/69c7c8fc5c0536077e0c6acf/1774700797001/Revista+Volumen+17.pdf)

---

## 📌 Features

* **Multi-Document RAG Architecture**: Loads and processes multiple PDF source files stored in the designated directory to answer complex regulatory questions.
* **Smart Context Chunking & Selection**: Uses `RecursiveCharacterTextSplitter` with token-based limits (`tiktoken`) to optimize token usage and avoid context window truncation.
* **Spanish Natural Language Processing**: Includes built-in text normalization, title handling, and conversational Spanish greeting detection.
* **Transparent Source Citation**: Displays exact document extracts used to compile each answer, complete with processing metrics.
* **Flexible OpenRouter Integration**: Connects to scalable OpenRouter models (`openrouter/auto`) for efficient completion generation.

---

## 📁 Directory Structure

```text
.
├── pdf_files_seguridad_social/     # Directory containing target IMSS PDF documents
├── app.py                          # Main Streamlit application file
├── ai-advisor-icon.svg             # UI Application logo
├── requirements.txt                # Python dependencies list
├── .env                            # Environment variables (API Keys)
└── README.md                       # Project documentation