# AI Chatbot Asesor de Pensiones IMSS 🇲🇽

> **Generative AI-Enabled Assistant for Mexican Social Security & Pension Queries**

A Streamlit-based web application designed to query, analyze, and synthesize official Mexican social security and IMSS (*Instituto Mexicano del Seguro Social*) pension documents using an advanced Retrieval-Augmented Generation (RAG) architecture powered by OpenRouter and LangChain.

* 🔗 **Live Application:** [AI Chatbot Asesor de Pensiones IMSS](https://chatbot-asesor-pensiones-imss.streamlit.app/)
* 🔗 **Read the article on Medium:** [Generative AI-Enabled Assistant for Pensions — Chatbot Asesor de Pensiones del IMSS](https://medium.com/latinxinai/generative-ai-enabled-assistant-for-pensions-a24435db6b01?sharedUserId=chomchom216)
* 🔗 **Read the article in Actuarios Trabajando (Mexico):** [Generative AI-Enabled Assistant for Pensions](https://static1.squarespace.com/static/694987771d60827a9dbf41de/t/69c7c8fc5c0536077e0c6acf/1774700797001/Revista+Volumen+17.pdf)

---

## 🌟 Key Features

* **Multi-Document RAG Architecture**: Loads, normalizes, and indexes PDF regulatory documents stored in the directory (`./pdf_files_seguridad_social`).
* **High-Availability OpenRouter Model Fallback**: Features automatic sequential model rotation across zero-cost inference endpoints on OpenRouter to protect against `429 Rate Limit Exceeded` errors and service outages:
  * `minimax/minimax-m3:free`
  * `google/gemma-4-31b:free`
  * `cohere/north-mini-code:free`
  * `openrouter/free-models-router` *(Managed Dynamic Router Fallback)*
* **Active Model Tracking & Response Streaming**: Streams completion tokens directly into the UI via `st.write_stream` and displays the exact active endpoint handling execution (`📝 Respuesta (Modelo activo: minimax/minimax-m3:free)`).
* **Intelligent Document Chunking**: Utilizes LangChain's `RecursiveCharacterTextSplitter` with `tiktoken` encoding to construct optimized context windows and prevent token truncation.
* **Spanish NLP & Greeting Handler**: Built-in time-aware greeting detection, accent normalization, and rule-based preprocessing for colloquial user inputs.
* **Streamlit Secrets Native Configuration**: Directly reads credentials from Streamlit Secrets portal (`st.secrets["OPENROUTER_API_KEY"]`), keeping repository deployments secure without requiring `.env` files.
* **Citation Transparency**: Expands source context blocks showing exact document chunks used to compile each query response.

---

## 🛠️ Tech Stack & Dependencies

* **Frontend / UI**: Streamlit (`>=1.30.0`)
* **LLM API Client**: Standard OpenAI SDK (`openai>=1.0.0`) via OpenRouter endpoint
* **Document Processing**: `langchain-community`, `langchain-text-splitters`, `pypdf`
* **Token Encoding & Utilities**: `tiktoken`

---

## ⚙️ Deployment & Streamlit Setup

### 1. Repository Setup
Push the application repository to GitHub:
```bash
git clone https://github.com/robert0777/chatbot-asesor-pensiones-imss.git
cd chatbot-asesor-pensiones-imss
```

### 2. Streamlit Cloud Secrets Setup
In your deployed app dashboard on Streamlit Cloud, navigate to **Settings** -> **Secrets**, and store your OpenRouter API key:
```toml
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```

### 3. Add Target PDF Documents
Place official IMSS regulatory and pension PDF files into:
```text
./pdf_files_seguridad_social/
```

### 4. Application Execution Entry
Point Streamlit Cloud deployment to:
```text
app3.3_spanish.py
```

---

## 📂 Project Structure

```text
.
├── app3.3_spanish.py               # Main Streamlit application entry point
├── pdf_files_seguridad_social/     # Directory for official IMSS PDF documents
├── ai-advisor-icon.svg             # Application logo & branding asset
├── requirements.txt                # Required Python dependencies
└── README.md                       # Project documentation
```

---

## 👤 Author

**Dr. Robert Hernández Martínez**  
*Consultant in Actuarial Science, Finance, Risk Modeling, and Applied AI*

* 📝 [Articles on Medium](https://chomchom216.medium.com/)
* 🎓 [Academic Publications](https://unam1.academia.edu/Robert_Hernandez_Martinez)
* 🏆 [Credentials on Credly](https://www.credly.com/users/robert-hernandez.89bffe7b)
* 🐙 [GitHub Profile](https://github.com/robert0777)
* 📧 Email: [robert@actuariayfinanzas.net](mailto:robert@actuariayfinanzas.net)
