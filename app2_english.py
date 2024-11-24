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
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./pdf_files")
        st.session_state.docs = st.session_state.loader.load()
        
        # Adjust chunk size to stay within token limit
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        # Create a vector store from the documents
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            
            # Print the number of documents processed for verification
            st.write(f"Processed {len(st.session_state.docs)} PDF files.")
            st.write(f"Created {len(st.session_state.final_documents)} document chunks.")
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            st.write("Try reducing the chunk size further if the error persists.")

st.title("Nvidia NIM Demo")

# Ensure base_url is correctly specified if needed
base_url = "https://integrate.api.nvidia.com/v1"  # Replace with actual base URL if necessary

try:
    llm = ChatNVIDIA(model="meta/llama3-70b-instruct", base_url=base_url)
except Exception as e:
    st.error(f"Error initializing ChatNVIDIA: {e}")
    st.stop()  # Stop execution if initialization fails

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    with st.spinner("Processing documents... This may take a while."):
        vector_embedding()
    st.success("Vector Store DB Is Ready")

if prompt1:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        
        # Prepare the payload as a dictionary
        payload = {'input': prompt1}
        response = retrieval_chain.invoke(payload)
        
        st.write(f"Response time: {time.process_time() - start} seconds")
        
        # Ensure response contains expected keys
        if 'answer' in response:
            st.write(response['answer'])
        else:
            st.write("No answer found in the response.")

        # Ensure 'context' is present and is iterable
        if 'context' in response and isinstance(response['context'], list):
            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
        else:
            st.write("No context found in the response.")
    except Exception as e:
        st.error(f"Error processing the request: {e}")
