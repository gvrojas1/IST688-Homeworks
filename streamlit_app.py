import streamlit as st
from openai import OpenAI

hw1 = st.Page("hw/hw1.py", title= "HW1")
hw2 = st.Page("hw/hw2.py", title= "HW2")
hw3 = st.Page("hw/hw3.py", title= "HW3")
hw4 = st.Page("hw/hw4.py", title= "HW4")
hw5 = st.Page("hw/hw5.py", title= "HW5")
hw7 = st.Page("hw/hw7.py", title= "HW7")

pg = st.navigation([hw1,hw2,hw3,hw4,hw5,hw7])
st.set_page_config(page_title="Homework Manager")
pg.run()
