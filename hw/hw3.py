import streamlit as st
from openai import OpenAI
from anthropic import Anthropic  
import requests
from bs4 import BeautifulSoup

# Function to read URL content 
def read_url_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        st.error(f"Error reading {url}: {e}")
        return None

BUFFER_SIZE = 6  #most recent messages (3 exchanges)

def build_buffer(full_history, summary):
    """
    Takes the FULL chat history and returns what to send to the LLM:
    the system prompt (URL context, never discarded) + a running summary of
    anything older than the buffer window + the last BUFFER_SIZE messages. 
    Combining the summary into the system message keeps this
    identical for OpenAI and Claude, since Claude requires a single system
    string rather than a separate system message.
    """
    system_msg = full_history[0]
    non_system_msgs = full_history[1:]
    recent_msgs = non_system_msgs[-BUFFER_SIZE:]
 
    if summary:
        combined_system = {
            "role": "system",
            "content": system_msg["content"]
            + f"\n\nSummary of earlier parts of this conversation:\n{summary}",
        }
        return [combined_system] + recent_msgs
 
    return [system_msg] + recent_msgs

def update_summary(llm_choice, openai_client, claude_client, previous_summary, new_messages):
    """Fold `new_messages` (about to fall out of the buffer) into a running summary."""
    convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in new_messages)
    prompt = (
        "Update the running summary of an ongoing conversation.\n\n"
        f"Current summary: {previous_summary if previous_summary else '(none yet)'}\n\n"
        f"New exchange to fold in:\n{convo_text}\n\n"
        "Return only the updated summary in 3-4 concise sentences, capturing "
        "key facts and context. No preamble."
    )
    messages = [{"role": "user", "content": prompt}]
 
    if llm_choice.startswith("OpenAI"):
        response = openai_client.chat.completions.create(model="gpt-6-astra", messages=messages)
        return response.choices[0].message.content
    else:
        response = claude_client.messages.create(model="claude-opus-5", max_tokens=300, messages=messages)
        for block in response.content:
            if block.type == "text":
                return block.text
        return previous_summary

def call_openai(client, api_messages):
    response = client.chat.completions.create(
        model="gpt-6-astra",
        messages=api_messages,
    )
    return response.choices[0].message.content
 
 
def call_claude(client, api_messages):
    # Anthropic's API takes the system prompt separately from the message list.
    system_msg = api_messages[0]["content"]
    convo_msgs = api_messages[1:]
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        system=system_msg,
        messages=convo_msgs,
    )
    for block in response.content:
        if block.type == "text":
            return block.text
    return ""

#Page set up 
# Show title and description.
st.title("Streaming Chatbot that discusses a URL")
st.write(
    "Enter a up to two URLs below, choose your LLM and start chatting.\n\n"
    "**How the memory works:** the text from your URL(s) is placed in the system "
    "prompt, which is never discarded, so the assistant always has that context. "
    "For the conversation itself, this uses a **buffer + summary** approach: the "
    "last 6 messages (3 exchanges) are sent to the LLM verbatim, and anything "
    "older than that is folded into a short running summary that's kept in the "
    "system prompt. Older messages stay visible on screen, but only the recent "
    "buffer and the summary are actually sent to the model."
)

#URls
st.sidebar.header("URLs")
url_1 = st.sidebar.text_input("URL 1")
url_2 = st.sidebar.text_input("URL 2")


#Sidebar: LLM selection
st.sidebar.header("LLM Options")
llm_choice = st.sidebar.selectbox(
    "Choose an LLM:",
    ("OpenAI (GPT-6 Astra)", "Claude (Opus 5)"),
)

#Built the system prompt from the URLs

url_contents = []
for url in (url_1, url_2):
    if url:
        content = read_url_content(url)
        if content:
            url_contents.append(f"Content from {url}:\n{content[:5000]}")
 
url_context = "\n\n---\n\n".join(url_contents) if url_contents else "No URL content provided yet."
 
SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a friendly assistant helping the user understand the following "
        f"web page content:\n\n{url_context}\n\n"
        "Answer the user's question clearly and simply using the content above. "
        "Then ALWAYS ask: 'Do you want more info?' "
        "If the user says yes, give more detail on the same topic, "
        "and then ask 'Do you want more info?' again. "
        "If the user says no, respond with 'Okay! What else can I help you with?'"
    ),
}

# Initialize OpenAI client

if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
 
if "claude_client" not in st.session_state:
    st.session_state.claude_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
 
#  FULL history — shown to the user 
 
if "messages" not in st.session_state:
    st.session_state.messages = [
        SYSTEM_PROMPT,
        {"role": "assistant", "content": "How can I help you?"},
    ]
else:
    # Keep the system prompt fresh if the URLs changed.
    st.session_state.messages[0] = SYSTEM_PROMPT
 
if "summary" not in st.session_state:
    st.session_state.summary = ""
 
if "summarized_up_to" not in st.session_state:
    st.session_state.summarized_up_to = 0  # how many non-system messages are already folded in
 
# Show FULL chat history on screen (but hide system message) 
 
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.write(message["content"])
 
prompt = st.chat_input("Type your message here...")
 
if prompt:
    # Add user message to the FULL history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
 
    # Build what to send to the LLM: system prompt + summary + recent buffer
    api_messages = build_buffer(st.session_state.messages, st.session_state.summary)
 
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if llm_choice.startswith("OpenAI"):
                answer = call_openai(st.session_state.openai_client, api_messages)
            else:
                answer = call_claude(st.session_state.claude_client, api_messages)
        st.write(answer)
 
    # Add the answer to the FULL history (for display)
    st.session_state.messages.append({"role": "assistant", "content": answer})
 
    # If new messages have fallen out of the buffer window, fold them into the summary
    non_system_msgs = st.session_state.messages[1:]
    cutoff = len(non_system_msgs) - BUFFER_SIZE
    if cutoff > st.session_state.summarized_up_to:
        new_msgs = non_system_msgs[st.session_state.summarized_up_to:cutoff]
        st.session_state.summary = update_summary(
            llm_choice,
            st.session_state.openai_client,
            st.session_state.claude_client,
            st.session_state.summary,
            new_msgs,
        )
        st.session_state.summarized_up_to = cutoff