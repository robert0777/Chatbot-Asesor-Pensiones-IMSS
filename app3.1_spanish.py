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
from streamlit.components.v1 import html



# Load the environment variables
load_dotenv()




# Set page configuration
st.set_page_config(
    layout="wide", 
    page_title="Consulta Seguridad Social",
    page_icon="📄"
)




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







# UI Enhancements
st.image("LOGOTIPO_ACTUARIA_FINANZAS-03.png", width=600)
st.title("Sistema de Consulta de Documentos de Seguridad Social Mexicana")
st.markdown("""
Este sistema analiza todos los documentos PDF en el directorio para proporcionar 
respuestas completas basadas en múltiples fuentes.
""")

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






















# def scroll_to_top_component():
    # CSS and JavaScript for the scroll-to-top functionality
#    st.markdown("""
#        <style>
#            #scrollToTopBtn {
#                position: fixed;
#                bottom: 30px;
#                right: 30px;
#                width: 40px;
#                height: 40px;
#                border-radius: 50%;
#                background-color: #0066cc;
#                color: white;
#                border: none;
#                font-size: 20px;
#                cursor: pointer;
#                display: flex;
#                align-items: center;
#                justify-content: center;
#                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
#                z-index: 999999;
#            }
            #scrollToTopBtn:hover {
#                background-color: #0052a3;
#            }
#        </style>
        
#        <button onclick="topFunction()" id="scrollToTopBtn" title="Go to top">↑</button>

#        <script>
#            function topFunction() {
#                document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
#                document.body.scrollTop = 0; // For Safari
#            }
            
#            // Show button when user scrolls down 20px
#            window.onscroll = function() {
#                scrollFunction()
#            };
            
#            function scrollFunction() {
#                const btn = document.getElementById("scrollToTopBtn");
#                if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
#                    btn.style.display = "flex";
#                } else {
#                    btn.style.display = "none";
#                }
#            }
            
#            // Initially hide the button
#            document.getElementById("scrollToTopBtn").style.display = "none";
#        </script>
#    """, unsafe_allow_html=True)


# scroll_to_top_component() 





    
    
# def scroll_to_top_component():
    # Create a container for the button in the sidebar
#    with st.sidebar:
        # Add some space before the button
#        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Create a button with a simple emoji as the label
#        if st.button("⬆️ Volver arriba", use_container_width=True):
            # Use JavaScript to scroll to top when button is clicked
#            js = '''
#                <script>
#                    window.scrollTo({top: 0, behavior: 'smooth'});
#                </script>
#            '''
#            st.components.v1.html(js, height=0)


# scroll_to_top_component() 





# Alternatively, you could place it in the main content area at the bottom
# def scroll_to_top_bottom():
    # Add some space before the button
#    st.markdown("<br>", unsafe_allow_html=True)
    
    # Center the button using columns
#    col1, col2, col3 = st.columns([1, 1, 1])
#    with col2:
#        if st.button("⬆️ Volver arriba", use_container_width=True):
#            js = '''
#                <script>
#                    window.scrollTo({top: 0, behavior: 'smooth'});
#                </script>
#            '''
#            st.components.v1.html(js, height=0)
# Place this before your footer
# scroll_to_top_bottom()










# def scroll_to_top_bottom():
#    st.markdown("""
#    <script>
#        function scrollToTop() {
#            window.scrollTo({ top: 0, behavior: 'smooth' });
#        }
#        document.getElementById("scroll-btn").onclick = scrollToTop;
#    </script>
#    <button id="scroll-btn" style="
#        position: fixed; bottom: 20px; right: 20px; 
#        background-color: #0066cc; color: white; 
#        border: none; border-radius: 50%; 
#        width: 40px; height: 40px; 
#        font-size: 20px; cursor: pointer;">
#        ↑
#    </button>
#    """, unsafe_allow_html=True)
# scroll_to_top_bottom()








# Hide Streamlit's default footer
hide_streamlit_style = """
<style>
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add custom footer
footer = """
<style>
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
"""
st.markdown(footer, unsafe_allow_html=True)





