import streamlit as st
import os
from openai import OpenAI  # Changed from langchain_nvidia_ai_endpoints
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Fixed import

# from langchain_core.prompts import ChatPromptTemplate
import time
from dotenv import load_dotenv
import tiktoken
from pathlib import Path
from functools import lru_cache
import re
import datetime

from langchain_nvidia_ai_endpoints import ChatNVIDIA  # Replace the OpenAI import


# Load environment variables
load_dotenv()

class GreetingHandler:
    def __init__(self):
        # Common Spanish greetings and pleasantries
        self.greetings = {
            # Basic greetings
            'hola', 'buenos días', 'buenas tardes', 'buenas noches', 'saludos',
            
            # Informal greetings
            'qué tal', 'cómo estás', 'como estas', 'qué hace', 'que hace',
            'qué onda', 'que onda',
            
            # Formal greetings
            'cómo le va', 'cómo está usted', 'como esta usted', 'mucho gusto',
            
            # Time-specific greetings
            'bonito día', 'feliz día', 'buen día',
            
            # Common variations
            'hey', 'hi', 'hello',
            
            # Introduction phrases
            'cómo te llamas', 'me llamo', 'mi nombre es',
            
            # Combined greetings
            'hola, qué tal', 'hola, cómo estás', 'hola, buenos días', 'hola, qué hace',
            
            # Regional variations
            'quiubo', 'qué hubo', 'que hubo', 'qué hay', 'que hay'
        }
        
        # Greeting responses based on time of day
        self.time_based_responses = {
            'morning': "¡Buenos días! ¿En qué puedo ayudarte con respecto a la seguridad social?",
            'afternoon': "¡Buenas tardes! ¿En qué puedo ayudarte con respecto a la seguridad social?",
            'evening': "¡Buenas noches! ¿En qué puedo ayudarte con respecto a la seguridad social?"
        }
        
        # Combined greeting patterns
        self.greeting_pattern = re.compile(
            '|'.join(r'\b{}\b'.format(re.escape(g)) for g in self.greetings),
            re.IGNORECASE
        )

    def normalize_text(self, text: str) -> str:
        """Normalize text by removing accents and converting to lowercase"""
        text = text.lower()
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ü': 'u', 'ñ': 'n'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def extract_question(self, text: str) -> str:
        """Remove greeting part from the text and return the actual question"""
        matches = list(self.greeting_pattern.finditer(text))
        if not matches:
            return text
            
        last_match_end = matches[-1].end()
        question = text[last_match_end:].strip(' ,.!?¿¡')
        return question if question else ""

    def process_input(self, user_input: str) -> tuple[bool, str | None, str | None]:
        """Process user input to handle greetings and questions"""
        if not user_input:
            return False, None, None
            
        normalized_input = self.normalize_text(user_input)
        is_greeting = bool(self.greeting_pattern.search(normalized_input))
        actual_question = self.extract_question(user_input)
        
        current_hour = datetime.datetime.now().hour
        if is_greeting:
            if current_hour < 12:
                greeting_response = self.time_based_responses['morning']
            elif current_hour < 18:
                greeting_response = self.time_based_responses['afternoon']
            else:
                greeting_response = self.time_based_responses['evening']
        else:
            greeting_response = None
            
        return is_greeting, greeting_response, actual_question

@lru_cache(maxsize=None)
def get_tokenizer():
    return tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def get_pdf_files(directory="./pdf_files_seguridad_social"):
    pdf_dir = Path(directory)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("No se encontraron archivos PDF en el directorio especificado")
    return pdf_files




def load_documents():
    if "documents" not in st.session_state:
        pdf_files = get_pdf_files()
        
        chunk_size = min(800, max(300, 500 * (50 / len(pdf_files))))  # Fixed this line
        chunk_overlap = max(50, chunk_size // 10)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=count_tokens,
            separators=["\n\n", "\n", ".", "!", "?", "¿", "¡", ";", ":", " ", ""]
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
    question_words = set(question.lower().split())
    chunk_words = set(chunk.page_content.lower().split())
    
    word_overlap = len(question_words.intersection(chunk_words))
    length_factor = 1 / (len(chunk.page_content.split()) + 1)
    
    return word_overlap * (1 - length_factor)

def normalize_spanish_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('D.', 'Doctor').replace('Dra.', 'Doctora')
    return text.strip()


# UI Setup
st.set_page_config(layout="wide", page_title="AI Chatbot Asesor de Pensiones IMSS", page_icon="❓")
st.image("ai-advisor-icon.svg", width=100)
st.header("Sistema de Consulta de Documentos de Seguridad Social Mexicana - IMSS")
st.markdown("Este sistema analiza todos los documentos PDF en el directorio para proporcionar respuestas completas basadas en múltiples fuentes.")

# Create custom CSS for the sidebar
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: white;
    }
    
    .sidebar-app-name {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1F2937;
    }
    
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #E5E7EB;
    }
    
    .sidebar-link {
        display: flex;
        align-items: center;
        color: #4B5563;
        text-decoration: none;
        padding: 0.5rem 0;
        transition: color 0.2s;
    }
    
    .sidebar-link:hover {
        color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    # App Logo and Title
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("ai-advisor-icon.svg", width=50)
    with col2:
        st.markdown('<p class="sidebar-app-name">AI Chatbot Asesor de Pensiones IMSS</p>', unsafe_allow_html=True)
    
    
    # About Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 📖 About this App")
    st.write("A powerful tool to analyze and query Mexican social security documents using advanced AI technology.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Author Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### 👤 Author")
    st.markdown("**Dr. Robert Hernández Martínez**")
    
    # Contact Links
    st.markdown("""
        <a href="https://chomchom216.medium.com/" class="sidebar-link">
            📝 Articles on Medium
        </a>
        <a href="https://unam1.academia.edu/Robert_Hernandez_Martinez" class="sidebar-link">
            🎓 Academic Publications
        </a>
        <a href="https://www.credly.com/users/robert-hernandez.89bffe7b" class="sidebar-link">
            🏆 Credentials
        </a>
        <a href="mailto:robert@actuariayfinanzas.net" class="sidebar-link">
            📧 Contact
        </a>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style="position: fixed; bottom: 0; padding: 1rem; text-align: center; font-size: 0.8rem; color: #6B7280;">
            © 2025 AI Chatbot Asesor de Pensiones IMSS
        </div>
    """, unsafe_allow_html=True)




# Replace your NVIDIA client initialization with this:

try:
    nvidia_client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Test connection with simpler parameters
    test_response = nvidia_client.chat.completions.create(
        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        messages=[
            {"role": "system", "content": "/think"},
            {"role": "user", "content": "test connection"}
        ],
        temperature=0.6,
        max_tokens=100,
        stream=False
    )
except Exception as e:
    st.error(f"""Failed to initialize NVIDIA NIMS client: {str(e)}
             Required actions:
             1. Verify API key in .env file
             2. Check model access at NGC dashboard
             3. Ensure account has NIMS permissions""")
    st.stop()







# Prompt template
prompt_template = """
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
"""

# Initialize greeting handler
if 'greeting_handler' not in st.session_state:
    st.session_state.greeting_handler = GreetingHandler()

# Input and Processing
prompt1 = st.text_input(
    "Introduzca su consulta:",
    placeholder="Su pregunta será analizada en todos los documentos disponibles"
)

if st.button("Click aquí para Cargar y Procesar Documentos"):
    with st.spinner('Cargando y procesando todos los documentos PDF...'):
        try:
            start_time = time.process_time()
            load_documents()
            processing_time = time.process_time() - start_time
            st.success(f"📚 Todos los documentos han sido cargados correctamente en {processing_time:.2f} segundos. ¡Ahora puede hacer sus preguntas!")
        except Exception as e:
            st.error(f"Error al cargar los documentos: {str(e)}")








# MODIFIED Query Processing section for NVIDIA NIMS:
if prompt1:
    is_greeting, greeting_response, actual_question = st.session_state.greeting_handler.process_input(prompt1)
    
    if is_greeting:
        st.write(greeting_response)
    
    if actual_question:
        if "documents" in st.session_state:
            try:
                with st.spinner('Analizando documentos...'):
                    start = time.process_time()
                    
                    selected_chunks = select_relevant_chunks(actual_question, st.session_state.documents)
                    
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
                    context = truncate_context(context)
                    
                    # Updated for NVIDIA NIMS model
                    completion = nvidia_client.chat.completions.create(
                        model="nvidia/llama-3.3-nemotron-super-49b-v1.5",
                        messages=[
                            {"role": "system", "content": "/think"},  # Required system prompt
                            {
                                "role": "user",
                                "content": prompt_template.format(
                                    context=context,
                                    question=actual_question
                                )
                            }
                        ],
                        temperature=0.6,  # Optimal for Spanish Q&A
                        top_p=0.95,
                        max_tokens=4000,  # Reduced from 65536 for practical usage
                        frequency_penalty=0,
                        presence_penalty=0,
                        stream=False  # Set to True if you want streaming
                    )
                    
                    st.write("📝 Respuesta:")
                    st.write(completion.choices[0].message.content)
                    
                    st.info(f"⏱️ Tiempo de procesamiento: {time.process_time() - start:.2f} segundos")
                    
                    st.write("\n📚 Documentos consultados:")
                    for doc_name, doc_chunks in docs_used.items():
                        with st.expander(f"Extractos de {doc_name}"):
                            for i, chunk in enumerate(doc_chunks, 1):
                                st.write(f"Extracto {i}:")
                                st.write(chunk)
                                st.markdown("---")
            
            except Exception as e:
                st.error(f"""Error durante el procesamiento: {str(e)}
                         Posibles soluciones:
                         1. Verifique su conexión a internet
                         2. Confirme el acceso al modelo en NGC
                         3. Pruebe con una consulta más corta""")
                
        else:
            st.warning("⚠️ Por favor, primero cargue los documentos usando el botón 'Cargar y Procesar Documentos'")
    elif not is_greeting:
        st.warning("Por favor, formule una pregunta específica sobre seguridad social.")





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
    Developed by Dr. Robert Hernández Martínez    |    robert@actuariayfinanzas.net    |    © 2025
</div>
""", unsafe_allow_html=True)
