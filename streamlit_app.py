import streamlit as st
from openai import OpenAI

hw1 = st.Page("hw/hw1.py", title= "HW1")
hw2 = st.Page("hw/hw2.py", title= "HW2")
hw3 = st.Page("hw/hw3.py", title= "HW3")

pg = st.navigation([hw1,hw2,hw3])
st.set_page_config(page_title="Homework Manager")
pg.run()
