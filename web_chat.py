import streamlit as st 
import logging 
import sys 
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. Configuration & setup 

st.set_page_config(page_title = 'My Local Llama RAG', layout = 'centered')
st.title('Chat with your Docs (Llama 3)')


# Initialize logging to see what's happening in the terminal 
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream = sys.stdout))

# 2. Cache the expensive steps (Loading data & model)
# We use @st.cache_resource so it only runs ONCE, not every time you click a button.
@st.cache_resource(show_spinner=False)
def load_data_and_model():
    with st.spinner(text="Loading models and indexing documents... This might take a minute."):
        
        # Setup Llama 3
        llm = Ollama(model='llama3', request_timeout=600.0)
        
        # Setup Embedding Model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Apply Settings
        Settings.llm = llm
        Settings.embed_model = embed_model
        
        # Load Documents
        # Make sure you have a folder named 'data' in the same directory
        documents = SimpleDirectoryReader(
            input_dir="./data", 
            required_exts=[".pdf", ".txt", ".md", ".docx"]
        ).load_data()
        
        # Create Index
        index = VectorStoreIndex.from_documents(documents)
        
        return index

# Load the index (this runs on first startup)
try:
    index = load_data_and_model()
    # Create the query engine
    if "query_engine" not in st.session_state:
        st.session_state.query_engine = index.as_query_engine()
    st.success("System Ready! Ask away.")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# 3. Chat Interface Logic

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I have read your documents. What would you like to know?"}
    ]

# Display all previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Your question here..."):
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.query_engine.query(prompt)
            st.markdown(response.response)
    
    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": response.response})
    

