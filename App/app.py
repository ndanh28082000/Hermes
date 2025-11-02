import streamlit as st
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.analytics import load_data
from Utils.query_parser import handle_query
from LLM.llm_handler import LlmHandler

st.set_page_config(page_title="Hermes", page_icon="🧠", layout="wide")

# Top header (title)
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.title("🧠 Hermes — AI Logistics Analyst")
with col2:
    st.caption("Compact chat + visual analytics")

# Initialize LLM handler once
if "llm_handler" not in st.session_state:
    st.session_state.llm_handler = LlmHandler()

# Session Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI settings in sidebar
with st.sidebar:
    st.header("Hermes Settings")
    if "chart_expanded" not in st.session_state:
        st.session_state.chart_expanded = False
    if "chart_width_px" not in st.session_state:
        st.session_state.chart_width_px = 800
    if "chart_height_px" not in st.session_state:
        st.session_state.chart_height_px = 300

    st.session_state.chart_expanded = st.checkbox("Expand charts by default", value=st.session_state.chart_expanded)
    st.session_state.chart_width_px = st.slider("Chart width (px)", 400, 1400, st.session_state.chart_width_px)
    st.session_state.chart_height_px = st.slider("Chart height (px)", 200, 900, st.session_state.chart_height_px)
    st.write("\n")
    st.markdown("**Quick prompts**")
    example_1 = st.button("Predict average delay next week")
    example_2 = st.button("Show the top 3 warehouses with the highest processing time")
    example_3 = st.button("Which route has the most delays last week?")

# Load data
df = load_data("D:/AI/Hermes/Data/shipments.csv")

# Layout: main chat column and a right-side panel for context / tips
main_col, right_col = st.columns([0.72, 0.28])

def _display_chat_history(container):
    for chat in st.session_state.chat_history:
        container.chat_message("user").write(chat["user"])
        # use markdown to preserve bold/formatting
        container.chat_message("assistant").write(chat["bot"])
        if chat.get("chart"):
            fig = chat["chart"]
            try:
                fig.set_size_inches(st.session_state.chart_width_px / fig.dpi, st.session_state.chart_height_px / fig.dpi)
            except Exception:
                pass
            with container.expander("Show chart", expanded=st.session_state.chart_expanded):
                container.pyplot(fig)

with main_col:
    chat_container = st.container()
    _display_chat_history(chat_container)

    # Chat input
    user_input = st.chat_input("Ask Hermes something...")

    # If user clicked quick examples, run them immediately as queries
    if example_1:
        user_input = "Predict average delay next week"
    if example_2:
        user_input = "Show the top 3 warehouses with the highest processing time"
    if example_3:
        user_input = "Which route has the most delays last week?"

    if user_input:
        context = "\n".join([f"User: {c['user']}\nHermes: {c['bot']}" for c in st.session_state.chat_history])
        chat_container.chat_message("user").write(user_input)

        with st.spinner("Hermes is analyzing your question..."):
            try:
                answer, fig = handle_query(user_input, df, st.session_state.llm_handler, chat_context=context)
            except Exception as e:
                answer = f"⚠️ Error while processing the query: {e}"
                fig = None

        # Show assistant reply (render markdown/formatting)
        chat_container.chat_message("assistant").write(answer)

        # Display chart in an expander sized according to sidebar settings
        if fig:
            try:
                fig.set_size_inches(st.session_state.chart_width_px / fig.dpi, st.session_state.chart_height_px / fig.dpi)
            except Exception:
                pass
            with chat_container.expander("Show chart", expanded=st.session_state.chart_expanded):
                chat_container.pyplot(fig)

        # Save history (store fig object for now)
        st.session_state.chat_history.append({"user": user_input, "bot": answer, "chart": fig})

with right_col:
    st.markdown("### Tips & Examples")
    st.write("Try asking Hermes in natural language. Examples:")
    st.write("- `Predict average delay next week`")
    st.write("- `Which warehouses have average delivery time above 5 days?`")
    st.write("- `List warehouses with average delivery time above 5 days`")
    st.markdown("---")
    st.write("You can control chart size and whether charts are expanded by default in the left sidebar.")
