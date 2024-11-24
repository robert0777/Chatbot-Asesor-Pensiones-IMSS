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

# Load the environment variables
load_dotenv()

# Initialize tokenizer
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    """Count tokens in text"""
    return len(tokenizer.encode(text))

def get_pdf_files():
    """List all PDF files in the directory"""
    pdf_dir = Path("./pdf_files_seguridad_social")
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise Exception("No se encontraron archivos PDF en el directorio especificado")
    return pdf_files

def load_documents():
    """Load and process PDF documents with detailed logging"""
    if "documents" not in st.session_state:
        st.session_state.document_info = {}
        pdf_files = get_pdf_files()
        
        # Log the files found
        st.write(f"Encontrados {len(pdf_files)} archivos PDF:")
        for pdf in pdf_files:
            st.write(f"- {pdf.name}")
        
 
        
        
        
        loader = PyPDFDirectoryLoader(
            path="./pdf_files_seguridad_social", 
            silent_errors=True,  # Ignore errors during PDF parsing
            recursive=False,     # Don't search subdirectories
            load_hidden=False    # Don't load hidden files
        )
        docs = loader.load()
        
        
        
        # Log the number of pages per document
        for doc in docs:
            filename = Path(doc.metadata['source']).name
            if filename not in st.session_state.document_info:
                st.session_state.document_info[filename] = {'pages': 0, 'chunks': 0}
            st.session_state.document_info[filename]['pages'] += 1
        
        # Optimize text splitting for Spanish content
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=count_tokens,
            separators=["\n\n", "\n", ".", "!", "?", "¿", "¡", ";", ":", " ", ""]
        )
        
        st.session_state.documents = text_splitter.split_documents(docs)
        
        # Count chunks per document
        for chunk in st.session_state.documents:
            filename = Path(chunk.metadata['source']).name
            st.session_state.document_info[filename]['chunks'] += 1
        
        # Display document statistics
        st.write("\nEstadísticas de procesamiento:")
        for filename, info in st.session_state.document_info.items():
            st.write(f"📄 {filename}:")
            st.write(f"   - Páginas: {info['pages']}")
            st.write(f"   - Fragmentos: {info['chunks']}")


        
  


def select_relevant_chunks(question, chunks, max_total_tokens=6000):
    """Select relevant chunks while ensuring representation from all documents"""
    prompt_tokens = count_tokens(question) + 500
    available_tokens = max_total_tokens - prompt_tokens
    
    # Group chunks by document
    docs_chunks = {}
    for chunk in chunks:
        doc_name = Path(chunk.metadata['source']).name
        if doc_name not in docs_chunks:
            docs_chunks[doc_name] = []
        docs_chunks[doc_name].append(chunk)
    
    # Select chunks from each document
    selected_chunks = []
    tokens_per_doc = available_tokens // len(docs_chunks)
    
    for doc_name, doc_chunks in docs_chunks.items():
        current_tokens = 0
        for chunk in doc_chunks:
            chunk_tokens = count_tokens(chunk.page_content)
            if current_tokens + chunk_tokens <= tokens_per_doc:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
    
    return selected_chunks

# UI Setup
st.title("Sistema de Consulta de Documentos de Seguridad Social Mexicana")
st.markdown("""
    Este sistema analiza todos los documentos PDF en el directorio para proporcionar 
    respuestas completas basadas en múltiples fuentes.
""")

# Initialize NVIDIA AI
base_url = "https://integrate.api.nvidia.com/v1"
try:
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct", base_url=base_url)
except Exception as e:
    st.error(f"Error al inicializar el modelo: {e}")
    st.stop()

# Optimized prompt for multi-document analysis
prompt = ChatPromptTemplate.from_template(
    """
    Eres un experto asistente especializado en el sistema de seguridad social mexicano. 
    Analiza cuidadosamente los siguientes extractos de MÚLTIPLES documentos oficiales para 
    proporcionar una respuesta completa y precisa.

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
)

            



# Input and document loading
prompt1 = st.text_input(
    "Introduzca su consulta:",
    placeholder="Su pregunta será analizada en todos los documentos disponibles"
)

if st.button("Cargar y Procesar Documentos"):
    # Use a spinner during document loading
    with st.spinner('Cargando y procesando todos los documentos PDF...'):
        try:
            load_documents()
            # Move the success message here, after loading is complete
            st.success("📚 Todos los documentos han sido cargados correctamente. ¡Ahora puede hacer sus preguntas!")
        except Exception as e:
            st.error(f"Error al cargar los documentos: {str(e)}")




# Query processing
if prompt1 and "documents" in st.session_state:
    try:
        with st.spinner('Analizando documentos...'):
            start = time.process_time()
            
            selected_chunks = select_relevant_chunks(prompt1, st.session_state.documents)
            
            # Group chunks by document for display
            docs_used = {}
            for chunk in selected_chunks:
                doc_name = Path(chunk.metadata['source']).name
                if doc_name not in docs_used:
                    docs_used[doc_name] = []
                docs_used[doc_name].append(chunk.page_content)
            
            # Combine selected chunks for context
            context_parts = []
            for doc_name, contents in docs_used.items():
                # Join contents first, then create the document section
                joined_contents = "\n".join(contents)
                doc_section = f"[Documento: {doc_name}]\n{joined_contents}"
                context_parts.append(doc_section)
            
            # Join all document sections with double newlines
            context = "\n\n".join(context_parts)
            
            response = llm.invoke(
                prompt.format_messages(
                    context=context,
                    question=prompt1
                )
            )
            
            # Display results
            st.write("📝 Respuesta:")
            st.write(response.content)
            st.info(f"⏱️ Tiempo de procesamiento: {time.process_time() - start:.2f} segundos")
            
            # Show source documents
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
