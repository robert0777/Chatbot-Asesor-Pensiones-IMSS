import streamlit as st
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

# Debugging: Check if the API key is loaded correctly
api_key = os.getenv("NVIDIA_API_KEY")
print(f"NVIDIA_API_KEY: {api_key}")  # This should print your API key
if api_key is None:
    raise ValueError("NVIDIA_API_KEY is not set. Please check your .env file.")

os.environ['NVIDIA_API_KEY'] = api_key

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings(model="nvolveqa_40k")  # Use a multilingual embedding model
        st.session_state.loader = PyPDFDirectoryLoader("./pdf_files_seguridad_social")
        st.session_state.docs = st.session_state.loader.load()
        
        # Ensure the documents are text-based and split correctly
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # Create a vector store from the documents
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        # Count the number of processed files
        st.session_state.num_files = len(set(doc.metadata.get('source') for doc in st.session_state.docs))

st.title("Demostración de Nvidia NIM para Documentos en Español")

# Ensure base_url is correctly specified if needed
base_url = "https://integrate.api.nvidia.com/v1"  # Replace with actual base URL if necessary

try:
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct", base_url=base_url)
    
except Exception as e:
    st.error(f"Error al inicializar ChatNVIDIA: {e}")
    st.stop()  # Stop execution if initialization fails

prompt = ChatPromptTemplate.from_template(
    """
    Responde a las preguntas basándote únicamente en el contexto proporcionado.
    Por favor, proporciona la respuesta más precisa basada en la pregunta.
    Asegúrate de considerar toda la información relevante de los diferentes documentos disponibles.
    <contexto>
    {context}
    </contexto>
    Pregunta: {input}
    """
)

prompt1 = st.text_input("Ingrese su pregunta sobre los documentos")

if st.button("Procesar Documentos"):
    vector_embedding()
    st.write(f"La base de datos vectorial está lista. Se procesaron {st.session_state.num_files} archivos PDF.")

if prompt1:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})  # Retrieve more documents
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        
        # Prepare the payload as a dictionary
        payload = {'input': prompt1}
        response = retrieval_chain.invoke(payload)
        
        st.write(f"Tiempo de respuesta: {time.process_time() - start} segundos")
        
        # Ensure response contains expected keys
        if 'answer' in response:
            st.write(response['answer'])
        else:
            st.write("No se encontró una respuesta en la respuesta.")

        # Ensure 'context' is present and is iterable
        if 'context' in response and isinstance(response['context'], list):
            with st.expander("Búsqueda de Similitud de Documentos"):
                for i, doc in enumerate(response['context']):
                    st.write(f"Documento: {doc.metadata.get('source', 'Desconocido')}")
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.write("No se encontró contexto en la respuesta.")
    except Exception as e:
        st.error(f"Error al procesar la solicitud: {e}")
