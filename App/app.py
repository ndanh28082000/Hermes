import streamlit as st
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.analytics import load_data
from Utils.query_parser import handle_query

st.set_page_config(page_title="Hermes", page_icon="🧠", layout="wide")
st.title("🧠 Hermes — AI Logistics Analyst")

# Session Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

df = load_data("D:/AI/Hermes/Data/shipments.csv")

# Display history
for chat in st.session_state.chat_history:
    st.chat_message("user").write(chat["user"])
    st.chat_message("assistant").write(chat["bot"])
    if chat.get("chart"):
        st.pyplot(chat["chart"])

user_input = st.chat_input("Ask Hermes something...")
if user_input:
    context = "\n".join([f"User: {c['user']}\nHermes: {c['bot']}" for c in st.session_state.chat_history])
    st.chat_message("user").write(user_input)

    with st.spinner("Hermes is analyzing your question..."):
        answer, fig = handle_query(user_input, df, chat_context=context)

    st.chat_message("assistant").write(answer)
    if fig:
        st.pyplot(fig)

    st.session_state.chat_history.append({"user": user_input, "bot": answer, "chart": fig})
