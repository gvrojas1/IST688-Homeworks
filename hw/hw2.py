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

# Show title and description.
st.title("#Url Summarizer")
st.write(
    "Enter a URL below and get an instant summary, choose the style, output language and LLM in the sidebar "
)

# URl input 
url = st.text_input("Enter a URL to summarize:")

#Sidebar-summary details
st.sidebar.header("Summary Options")
summary_type = st.sidebar.radio(
    "Choose a summary format:",
    (
        "Summarize in 100 words",
        "Summarize in 2 connecting paragraphs",
        "Summarize in 5 bullet points",
    ),
)

#Sidebar-output language 
language = st.sidebar.selectbox(
    "Choose output language:",
    ("English","Spanish","French")
)
use_advanced_model = st.sidebar.checkbox("Use advanced model")


#Sidebar: LLM selection
st.sidebar.header("LLM Options")
llm_choice = st.sidebar.selectbox(
    "Choose an LLM:",
    ("OpenAI","Claude"),
)

#set up client and validate keys
key_is_valid = False
client = None
model = None

if llm_choice == "OpenAI":
    openai_api_key = st.secrets.get("OPENAI_API_KEY","")
    client = OpenAI(api_key=openai_api_key)
    model = "gpt-4.1" if use_advanced_model else "gpt-4.1-mini"
    try:
        client.models.list()
        key_is_valid = True
    except Exception as e:
        st.error(f"Invalid OpenAI API Key: {e}")
elif llm_choice == "Claude":
    anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    client = Anthropic(api_key=anthropic_api_key)
    model = "claude-sonnet-4-6" if use_advanced_model else "claude-haiku-4-5-20251001"
    key_is_valid = bool(anthropic_api_key)  # just check it exists; real validation happens on first real call

# Main logic 
if key_is_valid and url:
    document = read_url_content(url)

    if document:
        instruction_map = {
            "Summarize in 100 words": "Summarize the content below in exactly 100 words.",
            "Summarize in 2 connecting paragraphs": "Summarize the content below in 2 connecting paragraphs.",
            "Summarize in 5 bullet points": "Summarize the content below in 5 concise bullet points.",
        }
        instruction = instruction_map[summary_type]

        prompt = (
            f"{instruction}\n\n"
            f"Write your entire response in {language}.\n\n"
            f"Here's the content:\n\n{document}"
        )

        if llm_choice == "OpenAI":
            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                st.write_stream(stream)
            except Exception as e:
                st.error(f"OpenAI API error: {e}")

        elif llm_choice == "Claude":
            try:
                with client.messages.stream(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    st.write_stream(stream.text_stream)
            except Exception as e:
                st.error(f"Claude API error: {e}")

elif not url:
    st.info("Please enter a URL above to get started.")
elif not key_is_valid:
    st.warning(f"Please provide a valid API key for {llm_choice}.")