import streamlit as st
from openai import OpenAI
import sys
import json
 
# A fix for working with ChromaDB on Streamlit
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules['pysqlite3']
except ImportError:
    pass
 
import chromadb
from pathlib import Path
from bs4 import BeautifulSoup
 
# Page setup
st.title("Student Organizations Chatbot (HW5 - with Tool Calling)")
 
# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
 
client = st.session_state.openai_client
 

# Helper functions used to build the vector DB

def extract_text_from_html(html_path):
    """Extracts the visible text from a 'Cuse Activities org page."""
    with open(html_path, encoding='utf-8', errors='ignore') as f:
        raw_html = f.read()
    soup = BeautifulSoup(raw_html, 'html.parser')
    for tag in soup(['script', 'style']):
        tag.decompose()
    text = soup.get_text(separator='\n')
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines)
 
def chunk_text(text):
    """Split each org page roughly in half by line count."""
    lines = text.split('\n')
    if len(lines) < 2:
        return [text, text]
    midpoint = len(lines) // 2
    return ['\n'.join(lines[:midpoint]), '\n'.join(lines[midpoint:])]
 
def add_to_collection(collection, text, file_name):
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks, start=1):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        collection.add(
            documents=[chunk],
            ids=[f"{file_name}_chunk{i}"],
            embeddings=[response.data[0].embedding]
        )
 
def load_htmls_into_collection(folder_path, collection):
    html_files = sorted(Path(folder_path).glob('*.html'))
    for html_path in html_files:
        text = extract_text_from_html(html_path)
        add_to_collection(collection, text, html_path.name)
 
def create_hw4_collection():
    """Reuses the same ChromaDB from HW4 (only builds it if empty)."""
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW4')
    collection = chroma_client.get_or_create_collection('HW4Collection')
    if collection.count() == 0:
        load_htmls_into_collection('./su_orgs', collection)
    return collection
 
def build_buffer(full_history, num_turns=5):
    """Short-term memory: system prompt + last `num_turns` exchanges."""
    system_msg = full_history[0]
    recent_msgs = full_history[1:][-(num_turns * 2):]
    return [system_msg] + recent_msgs
 
# Build (or reuse) the vector DB - same session key as HW4 so it isn't rebuilt
if 'HW4_VectorDB' not in st.session_state:
    with st.spinner("Building vector database from student organization pages..."):
        st.session_state.HW4_VectorDB = create_hw4_collection()
 
collection = st.session_state.HW4_VectorDB
 
# the tool the LLM can call

def relevant_club_info(query, n_results=5):
    """Takes a search query written by the LLM, runs a vector search
    on ChromaDB, and returns the matching org info as a string."""
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    results = collection.query(
        query_embeddings=[response.data[0].embedding],
        n_results=n_results
    )
    docs = results['documents'][0]
    ids = results['ids'][0]
    return "\n\n".join(f"--- From {ids[i]} ---\n{docs[i]}" for i in range(len(docs)))
 
# Tool description the LLM sees (so it knows when/how to call it)
tools = [
    {
        "type": "function",
        "function": {
            "name": "relevant_club_info",
            "description": (
                "Searches the Syracuse University student organization database "
                "and returns relevant excerpts from org pages. Use this whenever "
                "the user asks about a specific club, types of clubs, how to join, "
                "meeting times, contacts, or anything about SU student organizations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A short search query describing what info is needed, "
                                       "e.g. 'robotics or engineering clubs' or 'how to join the chess club'."
                    }
                },
                "required": ["query"]
            }
        }
    }
]
 
# Main App

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful advisor chatbot that helps Syracuse University students "
        "find and learn about student organizations ('Cuse Activities / Engage). "
        "You have a tool called relevant_club_info that searches the org database. "
        "Call it whenever the question is about SU student organizations. "
        "When you answer using the tool results, start with something like "
        "'Based on the organization info I found...'. "
        "For greetings or general questions that don't need org info, just answer "
        "directly without calling the tool. If the results aren't relevant, say so."
    )
}
 
# Used a different key than HW4 ("messages") so the two pages don't share chat history
if "hw5_messages" not in st.session_state:
    st.session_state.hw5_messages = [
        SYSTEM_PROMPT,
        {"role": "assistant", "content": "Hi! Ask me about Syracuse student organizations - "
                                          "what they do, how to join, or when they meet."}
    ]
 
# Show chat history (skip the system message)
for message in st.session_state.hw5_messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
prompt = st.chat_input("Ask about a student organization...")
 
if prompt:
    st.session_state.hw5_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
    api_messages = build_buffer(st.session_state.hw5_messages, num_turns=5)
 
    # STEP 1: Ask the LLM, giving it the option to call the tool
    first_response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=api_messages,
        tools=tools,
        tool_choice="auto",
    )
    response_msg = first_response.choices[0].message
 
    if response_msg.tool_calls:
        # STEP 2: The LLM decided to call the tool -> run the vector search
        api_messages.append({
            "role": "assistant",
            "content": response_msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in response_msg.tool_calls
            ],
        })
 
        for tc in response_msg.tool_calls:
            args = json.loads(tc.function.arguments)
            search_query = args.get("query", prompt)
            st.caption(f"🔎 Searching org database for: *{search_query}*")
            result = relevant_club_info(search_query)
 
            # Show the LLM the tool result directly
            api_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
 
        # STEP 3: Call the LLM again WITHOUT tools so it must answer from the results
        final_response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=api_messages,
        )
        answer = final_response.choices[0].message.content
    else:
        # The LLM didn't need the tool (general question)
        answer = response_msg.content
 
    # Only save the user/assistant text to history (keeps the memory buffer clean)
    st.session_state.hw5_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)