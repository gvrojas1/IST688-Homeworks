import streamlit as st
from openai import OpenAI
import sys

# A fix for working with ChromaDB on Streamlit
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass

import chromadb
from pathlib import Path
from bs4 import BeautifulSoup
import re

# Page setup
st.title("Student Organizations Chatbot")

#Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = st.session_state.openai_client

# Helper functions
# Extract text from html
def extract_text_from_html(html_path):
    """
    Extracts the visible text from a 'Cuse Activities org page to pass to
    add_to_collection() to add to the ChromaDB collection.
    """
    with open(html_path, encoding='utf-8', errors='ignore') as f:
        raw_html = f.read()
    soup = BeautifulSoup(raw_html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(separator='\n')
    # collapse blank lines to not embed a bunch of empty whitespace
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)

# Chunking method: fixed position split into two mini documents
# Split each org page roughly in half by line count to not cut a sentence in the middle 
def chunk_text(text):
    lines = text.split('\n')
    if len(lines) < 2:
        return [text, text]
    midpoint = len(lines) // 2
    chunk_1 = '\n'.join(lines[:midpoint])
    chunk_2 = '\n'.join(lines[midpoint:])
    return [chunk_1, chunk_2]

def add_to_collection(collection, text, file_name):
    # Split into two mini-documents per the assignment
    chunks = chunk_text(text)
    client = st.session_state.openai_client
    for i, chunk in enumerate(chunks, start=1):
        # Create an embedding
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        # Get the embedding vector
        embedding_vector = response.data[0].embedding
        # Add embedding and document to the collection (ChromaDB)
        collection.add(
            documents=[chunk],
            ids=[f"{file_name}_chunk{i}"],
            embeddings=[embedding_vector]
        )

# Populate collection with HTML org pages
def load_htmls_into_collection(folder_path, collection):
    """Reads all HTML files in folder_path and adds each one (as two chunks) to collection."""
    html_files = sorted(Path(folder_path).glob('*.html'))
    for html_path in html_files:
        text = extract_text_from_html(html_path)
        add_to_collection(collection, text, html_path.name)
 
def create_hw4_collection():
    """Creates (or opens) the HW4Collection ChromaDB collection and
    populates it from HTML org pages only if it's currently empty."""
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')
    collection = chroma_client.get_or_create_collection('HW4Collection')
 
    if collection.count() == 0:
        load_htmls_into_collection('./su_orgs', collection)
 
    return collection

def get_relevant_context(query, n_results=5):
    """Embeds the query, retrieves the top-n matching documents from the
    collection, and returns a combined context string plus the source filenames."""
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
 
    docs = results['documents'][0]
    ids = results['ids'][0]
    context_str = "\n\n".join(
        f"--- From {ids[i]} ---\n{docs[i]}" for i in range(len(docs))
    )
    return context_str, ids
 
 
def build_buffer(full_history, num_turns=5):
    """Returns the system prompt + only the last `num_turns` user/assistant
    exchanges, to keep token usage low."""
    system_msg = full_history[0]
    non_system_msgs = full_history[1:]
    recent_msgs = non_system_msgs[-(num_turns * 2):]
    return [system_msg] + recent_msgs

# Build (or reuse) the vector DB 
if 'HW4_VectorDB' not in st.session_state:
    with st.spinner("Building vector database from student organization pages..."):
        st.session_state.HW4_VectorDB = create_hw4_collection()

collection = st.session_state.HW4_VectorDB

    
#### Main App ####

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful advisor chatbot that helps Syracuse University students "
        "find and learn about student organizations ('Cuse Activities / Engage). "
        "You will sometimes be given relevant excerpts from organization pages under "
        "'RELEVANT ORG INFO'. Use that information to answer the user's question "
        "when it's relevant. If you used it, explicitly say something like "
        "'Based on the organization info I found...' at the start of your answer. "
        "If the retrieved info isn't relevant to the question, answer from general "
        "knowledge and say so instead."
    )
}
 
if "messages" not in st.session_state:
    st.session_state.messages = [
        SYSTEM_PROMPT,
        {"role": "assistant", "content": "Hi! Ask me about Syracuse student organizations - "
                                          "what they do, how to join, or when they meet."}
    ]
 
# Show chat history (skip the system message)
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
prompt = st.chat_input("Ask about a student organization...")
 
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
    # RAG retrieval for this turn 
    context_str, source_ids = get_relevant_context(prompt, n_results=5)
 
    api_messages = build_buffer(st.session_state.messages, num_turns=5)
    augmented_messages = api_messages + [
        {
            "role": "system",
            "content": f"RELEVANT ORG INFO (sources: {', '.join(source_ids)}):\n{context_str}"
        }
    ]
 
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=augmented_messages,
    )
    answer = response.choices[0].message.content
 
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
   