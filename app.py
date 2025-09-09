import os
import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import gdown
import zipfile
import shutil
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Atlan AI Copilot",
    page_icon="🤖",
    layout="wide"
)

# Load environment variables
load_dotenv()

# --- FUNCTION TO DOWNLOAD AND UNZIP THE INDEX ---
@st.cache_resource
def download_and_unzip_index():
    """
    Downloads the FAISS index from Google Drive and unzips it if not already present.
    """
    index_dir = "faiss_index"
    zip_file = "faiss_index.zip"

    if not os.path.exists(index_dir):
        file_id = st.secrets.get("GDRIVE_INDEX_ID")
        if not file_id:
            st.error("🚨 GDRIVE_INDEX_ID secret not found. Please add it in your Streamlit Cloud settings.")
            st.stop()

        # The spinner will appear only while this block is running
        with st.spinner("⏳ First-time setup: Downloading knowledge base... Please wait."):
            output = gdown.download(id=file_id, output=zip_file, quiet=False)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_file)

        st.success("✅ Knowledge base ready!")

# Helper function for colored status tags
def status_tag(text, color):
    return f'<span style="background-color:{color}; color:white; padding: 3px 10px; border-radius:15px; font-size:13px; margin: 2px;">{text}</span>'

# Cache file for persistent storage
CLASSIFICATION_CACHE_FILE = "classified_tickets_cache.json"

if not os.environ.get("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY is not set. Please add it to your .env file or Streamlit secrets.")
    st.stop()

# Download and unzip index if not present
download_and_unzip_index()

if not os.path.exists("data/sample_tickets.json"):
    st.error("❌ data/sample_tickets.json not found.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://assets-global.website-files.com/6145f4ba375a5e33a46a3628/650968988a623258c735d793_Atlan%20Logo%20(1).svg", width=150)
    st.header("AI Copilot")
    st.info("An AI-powered pipeline for classifying and responding to customer support tickets.")
    st.divider()
    st.header("Controls")
    if st.button("Clear Classification Cache"):
        if os.path.exists(CLASSIFICATION_CACHE_FILE):
            os.remove(CLASSIFICATION_CACHE_FILE)
            st.success("Cache cleared! App will re-classify on next run.")
            st.rerun()

# --- MAIN APPLICATION ---
st.title("🤖 Atlan AI Copilot")
st.caption("AI-powered ticket triage and automated support dashboard.")

# --- Cached Functions (for loading models and data) ---
@st.cache_data
def load_tickets():
    with open("data/sample_tickets.json", "r") as f:
        return json.load(f)

tickets = load_tickets()


# --- STEP 1: CACHE THE AI MODEL/CHAIN ---
@st.cache_resource
def get_cached_classification_chain():
    from modules.classification import get_classification_chain
    st.write("🚀 Initializing AI classification chain... (runs only once)")
    return get_classification_chain()




# --- STEP 2: PERSISTENT FILE-BASED CACHING ---
def load_or_classify_all_tickets(tickets_json, chain):
    """
    Loads classifications from a persistent file cache. If the cache doesn't exist,
    it classifies ALL tickets, shows a progress bar, and saves the result.
    """
    if os.path.exists(CLASSIFICATION_CACHE_FILE):
        st.info("✅ Loading pre-classified tickets from the local cache file.")
        with open(CLASSIFICATION_CACHE_FILE, 'r') as f:
            return json.load(f)

    st.warning("Cache file not found. Performing a one-time bulk classification of all tickets. This will take a few minutes...")
    results = []
    
    progress_bar = st.progress(0, text="Starting classification of all tickets...")

    total_tickets = len(tickets_json)
    for i, ticket in enumerate(tickets_json):
        # Respect per-minute rate limits (Gemini Free Tier is ~15 req/min)
        if i > 0:
            time.sleep(4)  # 4 seconds delay = 15 requests per minute

        try:
            ticket_text = f"{ticket['subject']}\n\n{ticket['body']}"
            classification = chain.invoke({"ticket_text": ticket_text})
            results.append({
                "id": ticket["id"], 
                "subject": ticket["subject"],
                "body": ticket["body"],
                "classification": classification.model_dump()
            })
        except Exception as e:
            st.error(f"An error occurred on ticket {ticket['id']}: {e}. Stopping.")
            break

        # Update the progress bar
        progress_percentage = (i + 1) / total_tickets
        progress_text = f"Classifying ticket {i+1}/{total_tickets}..."
        progress_bar.progress(progress_percentage, text=progress_text)

    # Save the results to the cache file for future runs
    with open(CLASSIFICATION_CACHE_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    progress_bar.empty()  # Remove the progress bar after completion
    st.success("✅ All tickets classified and saved to cache for instant loading next time.")
    return results


# --- SECTION 1: TEAM DASHBOARD ---
st.header("📊 Team Dashboard: Bulk Ticket Analysis")
st.caption("High-level overview of all support tickets, their classifications, and key trends.")

# First, get the cached chain. This will run only once.
classification_chain = get_cached_classification_chain()

with st.spinner("🤖 Loading or classifying tickets..."):
    classified_tickets = load_or_classify_all_tickets(tickets, classification_chain)

# --- Summary Metrics ---
total_tickets = len(classified_tickets)
high_priority = sum(1 for item in classified_tickets if "P0" in item['classification']['priority'])
frustrated_tickets = sum(1 for item in classified_tickets if "Frustrated" in item['classification']['sentiment'])

col1, col2, col3 = st.columns(3)
col1.metric("Total Tickets Analyzed", total_tickets)
col2.metric("High Priority (P0)", high_priority, delta=high_priority, delta_color="inverse")
col3.metric("Frustrated Customers", frustrated_tickets, delta=frustrated_tickets, delta_color="inverse")

# --- Analytics Charts ---
analytics_data = []
for item in classified_tickets:
    for topic in item['classification']['topic_tags']:
        analytics_data.append({
            'topic': topic,
            'sentiment': item['classification']['sentiment'],
            'priority': item['classification']['priority']
        })
analytics_df = pd.DataFrame(analytics_data)

chart_col1, chart_col2, chart_col3 = st.columns(3)
with chart_col1:
    topic_counts = analytics_df['topic'].value_counts()
    fig_topics = px.pie(
        values=topic_counts.values,
        names=topic_counts.index,
        title="Topic Distribution"
    )

    # This update moves the percentage to the hover tooltip
    fig_topics.update_traces(
        textinfo='none', # Hides the text on the pie slices
        hovertemplate='<b>Topic:</b> %{label}<br><b>Count:</b> %{value}<br><b>Percentage:</b> %{percent}'
    )
    st.plotly_chart(fig_topics, use_container_width=True)
with chart_col2:
    sentiment_counts = analytics_df['sentiment'].value_counts()
    fig_sentiment = px.bar(
        sentiment_counts,
        title="Sentiment Analysis",
        text_auto=True,
        labels={'index': 'sentiment', 'value': 'count'} # Renames the data for the tooltip
    )
    # This part customizes the text you see when you hover over a bar
    fig_sentiment.update_traces(
        hovertemplate='sentiment=%{x}<br>count=%{y}<extra></extra>'
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
with chart_col3:
    priority_counts = analytics_df['priority'].value_counts()
    fig_priority = px.bar(priority_counts, title="Priority Levels", text_auto=True)
    st.plotly_chart(fig_priority, use_container_width=True)

# --- Data Export ---
df_export = pd.DataFrame([{
    "Ticket_ID": item["id"], "Subject": item["subject"], "Topics": ", ".join(item['classification']['topic_tags']),
    "Sentiment": item['classification']['sentiment'], "Priority": item['classification']['priority'], "Body": item["body"]
} for item in classified_tickets])
st.download_button(
    label="📤 Export All Classified Tickets to CSV",
    data=df_export.to_csv(index=False).encode('utf-8'),
    file_name="atlan_classified_tickets.csv",
    mime="text/csv",
    use_container_width=True
)

st.divider()

# --- SECTION 2: DETAILED TICKET VIEW ---
st.header("📑 Detailed View: Classified Tickets")
st.caption("Review individual tickets and the AI's detailed classification for each.")

# Create two columns
col1, col2 = st.columns(2)

# Loop through the tickets with an index
for index, item in enumerate(classified_tickets):
    # Use the first column for even-indexed tickets
    if index % 2 == 0:
        with col1:
            with st.expander(f"**{item['id']}**: {item['subject']}"):
                classification = item['classification']
                
                # Display colored tags at the top
                priority_color = {"P0 (High)": "#D32F2F", "P1 (Medium)": "#F57C00", "P2 (Low)": "#388E3C"}.get(classification['priority'], "#757575")
                sentiment_color = {"Frustrated": "#D32F2F", "Angry": "red", "Curious": "#1976D2", "Neutral": "#757575"}.get(classification['sentiment'], "#757575")
                
                tags_html = status_tag(classification['priority'], priority_color)
                tags_html += status_tag(classification['sentiment'], sentiment_color)
                for topic in classification['topic_tags']:
                    tags_html += status_tag(topic, "#546E7A")
                
                st.markdown(tags_html, unsafe_allow_html=True)
                st.markdown("---")
                
                # Display ticket content
                st.markdown("**Ticket Content:**")
                st.text_area("", item['body'], height=150, disabled=True)
                
                with st.expander("View Raw Classification JSON"):
                    st.json(classification)

    # Use the second column for odd-indexed tickets
    else:
        with col2:
            with st.expander(f"**{item['id']}**: {item['subject']}"):
                classification = item['classification']
                
                # Display colored tags at the top
                priority_color = {"P0 (High)": "#D32F2F", "P1 (Medium)": "#F57C00", "P2 (Low)": "#388E3C"}.get(classification['priority'], "#757575")
                sentiment_color = {"Frustrated": "#D32F2F", "Angry": "red", "Curious": "#1976D2", "Neutral": "#757575"}.get(classification['sentiment'], "#757575")
                
                tags_html = status_tag(classification['priority'], priority_color)
                tags_html += status_tag(classification['sentiment'], sentiment_color)
                for topic in classification['topic_tags']:
                    tags_html += status_tag(topic, "#546E7A")
                
                st.markdown(tags_html, unsafe_allow_html=True)
                st.markdown("---")
                
                # Display ticket content
                st.markdown("**Ticket Content:**")
                st.text_area("", item['body'], height=150, disabled=True)
                
                with st.expander("View Raw Classification JSON"):
                    st.json(classification)

st.divider()
# --- SECTION 3: INTERACTIVE SIMULATION ---
st.header("� Interactive Copilot Simulation")
st.caption("Choose a view to test the AI copilot's real-time analysis and response capabilities.")

# --- TABS for Interface Selection ---
tab1, tab2 = st.tabs(["🧑‍� Triage Simulation (Support Team's View)", "� Live Chat Simulation (Customer's View)"])

# --- TAB 1: TRIAGE SIMULATION ---
with tab1:
    st.subheader("Submit a Ticket to See the AI's Internal Analysis")
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    user_query = st.text_area(
        "Describe the issue or ask a question:",
        placeholder="Example: How do I set up SSO with Azure AD?",
        height=100, key="triage_input"
    )

    if st.button("🚀 Analyze Ticket", key="triage_submit", type="primary"):
        if user_query.strip():
            with st.spinner("🧠 Analyzing your query..."):
                analysis = classification_chain.invoke({"ticket_text": user_query})
                topic_tags = [t.lower() for t in analysis.topic_tags]
                rag_topics = ["how-to", "product", "best practices", "api/sdk", "sso"]

                if any(t in rag_topics for t in topic_tags):
                    from modules.rag import get_rag_response_stream
                    
                    # Build conversation history from recent query history for context
                    conversation_history = ""
                    if len(st.session_state.query_history) > 0:  # If we have previous queries
                        recent_queries = st.session_state.query_history[:2]  # Get last 2 queries
                        for q in reversed(recent_queries):  # Reverse to show chronological order
                            conversation_history += f"user: {q['query']}\nassistant: {q['answer']}\n"
                    
                    full_response, sources = "", []
                    for part in get_rag_response_stream(user_query, conversation_history):
                        if "chunk" in part: full_response += part["chunk"]
                        elif "sources" in part: sources = part["sources"]
                    answer = full_response
                else:
                    answer = f"📨 This ticket has been classified as a **'{analysis.topic_tags[0]}'** issue and has been automatically routed to the appropriate specialist team based on its **{analysis.priority}** priority level."
                    sources = []
                
                st.session_state.query_history.insert(0, {
                    "query": user_query, "analysis": analysis.model_dump(),
                    "answer": answer, "sources": sources
                })
        else:
            st.warning("Please enter a query.")

    if st.session_state.query_history:
        last = st.session_state.query_history[0]
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("🔍 Internal Analysis (Back-end)")
            st.markdown(f"**Topics:** {', '.join(last['analysis']['topic_tags'])}")
            st.markdown(f"**Sentiment:** {last['analysis']['sentiment']}")
            st.markdown(f"**Priority:** {last['analysis']['priority']}")
            with st.expander("Raw Analysis JSON"):
                st.json(last['analysis'])
        with res_col2:
            st.subheader("💡 Final Response (Front-end)")
            st.markdown(last['answer'])
            if last['sources']:
                st.markdown("**🔗 Sources:**")
                for url in last['sources']:
                    st.markdown(f"- [{url}]({url})")

# --- TAB 2: LIVE CHAT SIMULATION ---
with tab2:
    st.subheader("Chat with the AI Assistant in Real-Time")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                st.info(f"**Sources:**\n" + "\n".join(f"- {source}" for source in message["sources"]))

    if prompt := st.chat_input("Ask a question about Atlan..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response, sources = "", []
            try:
                analysis = classification_chain.invoke({"ticket_text": prompt})
                topic_tags = [t.lower() for t in analysis.topic_tags]
                rag_topics = ["how-to", "product", "best practices", "api/sdk", "sso", "glossary", "lineage", "connector"]

                if any(t in rag_topics for t in topic_tags):
                    from modules.rag import get_rag_response_stream
                    
                    # Build conversation history from recent messages for context
                    conversation_history = ""
                    if len(st.session_state.messages) > 1:  # If we have previous messages
                        recent_messages = st.session_state.messages[-4:]  # Get last 4 messages (2 turns)
                        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_messages])
                    
                    for part in get_rag_response_stream(prompt, conversation_history):
                        if "chunk" in part:
                            full_response += part["chunk"]
                            response_placeholder.markdown(full_response + "▌")
                        elif "sources" in part: sources = part["sources"]
                    response_placeholder.markdown(full_response)
                else:
                    full_response = f"This has been classified as **'{', '.join(analysis.topic_tags)}'** with **{analysis.priority} priority**. It has been routed to the appropriate team for handling."
                    response_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"🚨 Apologies, an error occurred: {str(e)}"
                response_placeholder.error(full_response)
            
            assistant_message = {"role": "assistant", "content": full_response}
            if sources:
                assistant_message["sources"] = sources
                st.info(f"**Sources:**\n" + "\n".join(f"- {source}" for source in sources))
            st.session_state.messages.append(assistant_message)