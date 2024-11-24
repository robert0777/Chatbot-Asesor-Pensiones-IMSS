import streamlit as st
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
import time
from dotenv import load_dotenv
import tiktoken
from pathlib import Path
from functools import lru_cache
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(layout="wide", page_title="Consulta Seguridad Social", page_icon="📄")

# Cached tokenizer
@lru_cache(maxsize=None)
def get_tokenizer():
    return tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    """Count tokens in text with cached tokenizer"""
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def get_pdf_files(directory="./pdf_files_seguridad_social"):
    """List all PDF files in the directory, sorted by name"""
    pdf_dir = Path(directory)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No se encontraron archivos PDF en el directorio especificado")
    return pdf_files

def load_documents():
    """Enhanced document loading with advanced chunking"""
    if "documents" not in st.session_state:
        pdf_files = get_pdf_files()
        
        chunk_size = min(800, max(300, 500 * (50 / len(pdf_files))))
        chunk_overlap = max(50, chunk_size // 10)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
            separators=[
                "\n\n", "\n", ".", "!", "?", 
                "¿", "¡", ";", ":", " ", ""
            ]
        )
        
        loader = PyPDFDirectoryLoader(
            path="./pdf_files_seguridad_social", 
            silent_errors=True,
            recursive=False
        )
        docs = loader.load()
        
        processed_docs = []
        for doc in docs:
            normalized_text = normalize_spanish_text(doc.page_content)
            doc.page_content = normalized_text
            processed_docs.append(doc)
        
        st.session_state.documents = text_splitter.split_documents(processed_docs)

def select_relevant_chunks(question, chunks, max_total_tokens=6000):
    """Strictly controlled chunk selection with token limit management"""
    prompt_tokens = count_tokens(question) + 500
    available_tokens = max_total_tokens - prompt_tokens
    
    scored_chunks = []
    for chunk in chunks:
        relevance_score = calculate_chunk_relevance(chunk, question)
        scored_chunks.append((chunk, relevance_score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    
    selected_chunks = []
    used_documents = set()
    current_tokens = 0
    
    for chunk, score in scored_chunks:
        doc_name = Path(chunk.metadata['source']).name
        chunk_tokens = count_tokens(chunk.page_content)
        
        if (doc_name not in used_documents and 
            current_tokens + chunk_tokens <= available_tokens):
            selected_chunks.append(chunk)
            used_documents.add(doc_name)
            current_tokens += chunk_tokens
        
        if current_tokens >= available_tokens * 0.9:
            break
    
    return selected_chunks

def truncate_context(context, max_tokens=6000):
    """Ensure context stays within token limits"""
    tokens = count_tokens(context)
    if tokens > max_tokens:
        lines = context.split('\n')
        truncated_context = []
        current_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if current_tokens + line_tokens <= max_tokens:
                truncated_context.append(line)
                current_tokens += line_tokens
            else:
                break
        
        return '\n'.join(truncated_context)
    return context

def calculate_chunk_relevance(chunk, question):
    """Calculate chunk relevance using basic semantic matching"""
    question_words = set(question.lower().split())
    chunk_words = set(chunk.page_content.lower().split())
    
    word_overlap = len(question_words.intersection(chunk_words))
    length_factor = 1 / (len(chunk.page_content.split()) + 1)
    
    return word_overlap * (1 - length_factor)

def normalize_spanish_text(text):
    """Basic Spanish text normalization"""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('D.', 'Doctor').replace('Dra.', 'Doctora')
    return text.strip()

# UI Setup
st.image("LOGOTIPO_ACTUARIA_FINANZAS-03.png", width=600)
st.title("Sistema de Consulta de Documentos de Seguridad Social Mexicana")
st.markdown("Este sistema analiza todos los documentos PDF en el directorio para proporcionar respuestas completas basadas en múltiples fuentes.")

with st.sidebar:
    st.image("LOGOTIPO_ACTUARIA_FINANZAS-03.png", width=300)
    st.markdown("## 📖 About This App")
    st.write("Analyze and query Mexican social security documents.")
    st.markdown("""
    ---
    **Author**: [Dr. Robert Hernández Martínez](https://www.credly.com/users/robert-hernandez.89bffe7b)  
    📧 [Contact](mailto:robert@actuariayfinanzas.net)  
    🌐 [Articles 01](https://chomchom216.medium.com/)
    🌐 [Articles 02](https://unam1.academia.edu/Robert_Hernandez_Martinez)
    ---
    """)

# Initialize NVIDIA AI
base_url = "https://integrate.api.nvidia.com/v1"
try:
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct", base_url=base_url)
except Exception as e:
    st.error(f"Error al inicializar el modelo: {e}")
    st.stop()




prompt = ChatPromptTemplate.from_template("""
Basado en la consulta específica sobre "{question}", analiza cuidadosamente 
los siguientes extractos de documentos oficiales de seguridad social mexicana 
para proporcionar una respuesta completa y precisa.

Instrucciones específicas:
1. Utiliza información de TODOS los documentos relevantes proporcionados.
2. Cuando encuentres información complementaria en diferentes documentos, 
   combínala de manera coherente.
3. Si hay discrepancias entre documentos, menciónalas.
4. Cita específicamente de qué documento proviene cada parte de tu respuesta.
5. Si la información está incompleta, indica qué documentos adicionales podrían ser necesarios.

Extractos de los documentos:
{context}

Pregunta: {question}

Respuesta (basada en múltiples documentos):
""")






# Input and Processing
prompt1 = st.text_input(
    "Introduzca su consulta:",
    placeholder="Su pregunta será analizada en todos los documentos disponibles"
)

if st.button("Cargar y Procesar Documentos"):
    with st.spinner('Cargando y procesando todos los documentos PDF...'):
        try:
            load_documents()
            st.success("📚 Todos los documentos han sido cargados correctamente. ¡Ahora puede hacer sus preguntas!")
        except Exception as e:
            st.error(f"Error al cargar los documentos: {str(e)}")

# Query Processing
if prompt1 and "documents" in st.session_state:
    try:
        with st.spinner('Analizando documentos...'):
            start = time.process_time()
            
            selected_chunks = select_relevant_chunks(prompt1, st.session_state.documents)
            
            docs_used = {}
            for chunk in selected_chunks:
                doc_name = Path(chunk.metadata['source']).name
                if doc_name not in docs_used:
                    docs_used[doc_name] = []
                docs_used[doc_name].append(chunk.page_content)
            
            context_parts = []
            for doc_name, contents in docs_used.items():
                joined_contents = "\n".join(contents)
                doc_section = f"[Documento: {doc_name}]\n{joined_contents}"
                context_parts.append(doc_section)
            
            context = "\n\n".join(context_parts)
            context = truncate_context(context)  # New token limit management
            
            response = llm.invoke(
                prompt.format_messages(
                    context=context,
                    question=prompt1
                )
            )
            
            st.write("📝 Respuesta:")
            st.write(response.content)
            st.info(f"⏱️ Tiempo de procesamiento: {time.process_time() - start:.2f} segundos")
            
            st.write("\n📚 Documentos consultados:")
            for doc_name, doc_chunks in docs_used.items():
                with st.expander(f"Extractos de {doc_name}"):
                    for i, chunk in enumerate(doc_chunks, 1):
                        st.write(f"Extracto {i}:")
                        st.write(chunk)
                        st.markdown("---")
    
    except Exception as e:
        st.error(f"Error durante el procesamiento: {str(e)}")
        
elif prompt1:
    st.warning("⚠️ Por favor, primero cargue los documentos usando el botón 'Cargar y Procesar Documentos'")

# Footer
st.markdown("""
<style>
footer {visibility: hidden;}
div.custom-footer {
    text-align: center;
    padding: 10px;
    font-size: 14px;
    color: gray;
    margin-top: 50px;
}
</style>
<div class="custom-footer">
    Developed by Dr. Robert Hernández Martínez    |    robert@actuariayfinanzas.net    |    © 2024
</div>
""", unsafe_allow_html=True)