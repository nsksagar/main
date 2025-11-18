import sys
import logging 

# This turn on logging so you can see what the program is doing 

logging.basicConfig(stream = sys.stdout, level = logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama

#---- Step.1 Load your documents ---- 

print('loading documents from "./data"')

# --- BEST PRACTICE: Added required_exts to only load files you want ---
# --- Change this list to match your files! ---
documents = SimpleDirectoryReader(
    input_dir="./data", 
    required_exts=[".pdf", ".txt", ".md", ".docx"] 
).load_data()

print(f'Loaded {len(documents)} documents')

# Step 2. Connect this to the brain 

# This connects to the llama3 
llm = Ollama(model = 'llama3', request_timeout=600.0)
# We are now telling LlamaIndex what to use by default.
Settings.llm = llm
Settings.embed_model = "local:BAAI/bge-small-en-v1.5" # <-- This tells it to use a free, local model

# Step 3. Create Knowledge base index 

#This reads your documents and create searchable vector indexes 
print('Creating Index')
Index = VectorStoreIndex.from_documents(documents)
print('Index Created Successfully')

# Step 4. Create the query engine 

# This object will answer your questions 

query_engine = Index.as_query_engine() # No need to pass llm here, Settings handles it

# Step 5. Create a loop to answer your question 
while True:
    question = input('Your Question: ')


    if question.lower() == 'exit':
        break


    response = query_engine.query(question)
    print(f'\nAI Answer: {response}\n')