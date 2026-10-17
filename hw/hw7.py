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

# Page setup
st.title("HW7: News Information Bot")
st.caption("Reports only on the news articles in our client news file.")

# Model choice: lower-cost vs higher-cost LLM

MODELS = {
    "gpt-5-nano (lower cost)": "gpt-5-nano",
    "gpt-5 (higher cost)": "gpt-5",
}
model_label = st.sidebar.selectbox("Which LLM?", list(MODELS.keys()))
MODEL = MODELS[model_label]
top_k = st.sidebar.slider("Articles in a ranked list", 3, 15, 10)

if st.sidebar.button("Clear chat"):
    st.session_state.pop("hw7_messages", None)

# Create OpenAI client
if 'openai_client' not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
client = st.session_state.openai_client

# Load the vector DB that build_news_db.py created 

@st.cache_resource
def load_news_collection():
    chroma_client = chromadb.PersistentClient(path='./ChromaDB_for_HW7')
    return chroma_client.get_or_create_collection('HW7_News')

collection = load_news_collection()

if collection.count() == 0:
    st.error("The news database is empty. Run `python build_news_db.py` first, "
             "then commit the ChromaDB_for_HW7 folder to GitHub.")
    st.stop()

@st.cache_data
def load_all_articles():
    """Every article + its metadata, used for ranking the whole news file."""
    data = collection.get(include=["documents", "metadatas"])
    return [{"id": i, "document": d, **m}
            for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])]

ALL_ARTICLES = load_all_articles()
COMPANIES = sorted({a["company"] for a in ALL_ARTICLES})
NEWEST_DAY = max(a["days_since_2000"] for a in ALL_ARTICLES)

# Helper functions

def match_company(name):
    """Turn what the LLM wrote (e.g. 'jpmorgan') into the exact company name in our data."""
    if not name:
        return None
    name = name.lower().strip()
    for c in COMPANIES:
        if c.lower() == name:
            return c
    for c in COMPANIES:
        if name in c.lower() or c.lower() in name:
            return c
    return None   # not a company in our file -> fall back to plain semantic search

def vector_search(query, company=None, n_results=8):
    """Embed the query and search ChromaDB (optionally only one company's articles)."""
    emb = client.embeddings.create(input=query, model="text-embedding-3-small")
    results = collection.query(
        query_embeddings=[emb.data[0].embedding],
        n_results=n_results,
        where={"company": company} if company else None,
    )
    return [{"id": i, "document": d, **m}
            for i, d, m in zip(results["ids"][0], results["documents"][0], results["metadatas"][0])]

def interest_score(article):
    """Legal/business score (computed in build_news_db.py) + up to 2 points for recency."""
    recency = 1 - (NEWEST_DAY - article["days_since_2000"]) / 7
    return article["legal_score"] + 2 * max(recency, 0)

def format_articles(articles, show_score=False):
    lines = []
    for n, a in enumerate(articles, 1):
        text = a["document"][:700]
        line = f"[{n}] Company: {a['company']} | Date: {a['date']} | URL: {a['url']}"
        if show_score:
            line += f" | Interest score: {interest_score(a):.1f} | Signals: {a['categories']}"
        lines.append(line + f"\n{text}")
    return "\n\n".join(lines) if lines else "No matching articles found in the news file."

def build_buffer(full_history, num_turns=5):
    """Short-term memory: system prompt + last `num_turns` exchanges (same as HW4/HW5)."""
    return [full_history[0]] + full_history[1:][-(num_turns * 2):]


# two tools the LLM can call

def find_news(query, company=None):
    """'Find news about X' -> semantic search, filtered to a company if one is named."""
    matched = match_company(company)
    articles = vector_search(query, company=matched, n_results=8)
    return format_articles(articles)

def rank_interesting_news(topic=None, company=None):
    """'Find the most interesting news' -> score candidates and return them best-first."""
    matched = match_company(company)
    if topic:
        candidates = vector_search(topic, company=matched, n_results=40)
    elif matched:
        candidates = [a for a in ALL_ARTICLES if a["company"] == matched]
    else:
        candidates = ALL_ARTICLES
    ranked = sorted(candidates, key=interest_score, reverse=True)[:25]
    return format_articles(ranked, show_score=True)

TOOL_FUNCTIONS = {"find_news": find_news, "rank_interesting_news": rank_interesting_news}

tools = [
    {
        "type": "function",
        "function": {
            "name": "find_news",
            "description": "Search the client news file for articles about a specific topic or company. "
                           "Use for requests like 'find news about X' or 'what's happening with X'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for, e.g. 'Zelle payments regulation'."},
                    "company": {"type": "string", "description": "Company name if the user named one, e.g. 'JPMorgan Chase'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_interesting_news",
            "description": "Return candidate articles pre-ranked by an interest score (legal/regulatory, "
                           "M&A, crisis, leadership and financial signals plus recency). Use for "
                           "'most interesting', 'most important', 'top stories' or 'what should we know' requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Optional topic to narrow the ranking, e.g. 'antitrust'."},
                    "company": {"type": "string", "description": "Optional company to narrow the ranking."},
                },
            },
        },
    },
]


# Main App

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a news monitoring assistant for a large global law firm that tracks news about its clients. "
        "You may ONLY report on articles returned by your tools. Never use outside knowledge about the news, "
        "and if the tools return nothing relevant, say the news file has nothing on that.\n\n"
        "When the user asks for the most interesting or important news, call rank_interesting_news. "
        f"Then return a numbered list of the top {top_k} stories, best first. The interest score is a "
        "starting point, not the final word: drop articles that are off-topic or not really about the company "
        "(for example a car article that only matches 'HP' because of horsepower), and merge several articles "
        "about the same story into one item. For each item give: the headline, company, date, the URL, and "
        "1-2 sentences on WHY it matters to a law firm (legal exposure, regulatory risk, deal activity, "
        "reputational risk, or business impact for the client).\n\n"
        "When the user asks for news about a topic or company, call find_news and summarize the relevant "
        "articles with dates and URLs. For greetings or questions about what you can do, answer directly."
    ),
}

if "hw7_messages" not in st.session_state:
    st.session_state.hw7_messages = [
        SYSTEM_PROMPT,
        {"role": "assistant", "content": "Hi! Ask me for the most interesting news, or for news about a "
                                          "specific company or topic."},
    ]
st.session_state.hw7_messages[0] = SYSTEM_PROMPT   # keep system prompt in sync with the slider

for message in st.session_state.hw7_messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("e.g. Find the most interesting news / Find news about Nvidia")

if prompt:
    st.session_state.hw7_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = build_buffer(st.session_state.hw7_messages, num_turns=5)

    with st.spinner(f"Thinking with {MODEL}..."):
        # STEP 1: Ask the LLM, giving it the option to call a tool
        first_response = client.chat.completions.create(
            model=MODEL,
            messages=api_messages,
            tools=tools,
            tool_choice="auto",
        )
        response_msg = first_response.choices[0].message

        if response_msg.tool_calls:
            # STEP 2: Run whichever tool(s) the LLM asked for
            api_messages.append({
                "role": "assistant",
                "content": response_msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in response_msg.tool_calls
                ],
            })
            for tc in response_msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                st.caption(f"🔎 {tc.function.name}({args})")
                func = TOOL_FUNCTIONS[tc.function.name]
                # only pass the arguments this function actually accepts
                allowed = func.__code__.co_varnames[:func.__code__.co_argcount]
                result = func(**{k: v for k, v in args.items() if k in allowed})
                api_messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            # STEP 3: Call again WITHOUT tools so it must answer from the results
            final_response = client.chat.completions.create(model=MODEL, messages=api_messages)
            answer = final_response.choices[0].message.content
        else:
            answer = response_msg.content

    st.session_state.hw7_messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)