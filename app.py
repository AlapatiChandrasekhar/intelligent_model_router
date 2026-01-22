import streamlit as st
import logic
import json
import os

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
st.set_page_config(layout="wide", page_title="Intelligent Model Router")

# Egnyte Colors
EGNYTE_BLUE = "#007BFF"
EGNYTE_GREEN = "#28A745"

# Custom CSS for styling headers, metrics, and containers
st.markdown(f"""
    <style>
    h1, h2, h3 {{
        color: {EGNYTE_BLUE};
    }}
    .stMetric > div > div > div > div {{
        color: {EGNYTE_GREEN};
    }}
    [data-testid="stSidebar"] {{
        background-color: #262730;
        color: white;
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    </style>
    """, unsafe_allow_html=True)

STATS_FILE = "stats.json"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_stats():
    """Loads stats from JSON file. Returns default structure if file not found or error."""
    default_stats = {
        "total_time_saved": 0.0,
        "total_cost_saved": 0.0,
        "total_carbon_saved": 0.0,
        "total_tokens_saved": 0,
        "last_session": {
            "model_name": "N/A",
            "cost_saved": 0.0,
            "time_saved": 0.0,
            "carbon_footprint": 0.0,
            "tokens_used": 0
        }
    }
    
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return default_stats
    return default_stats

def save_stats(stats):
    """Saves stats dictionary to JSON file."""
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)

def update_stats(new_metrics):
    """Updates global accumulators and last session data."""
    stats = load_stats()
    
    # Update Totals
    stats["total_time_saved"] += new_metrics["time_saved"]
    stats["total_cost_saved"] += new_metrics["cost_saved"]
    stats["total_carbon_saved"] += new_metrics["carbon_footprint"]
    stats["total_tokens_saved"] += new_metrics["tokens_used"]
    
    # Update Last Session
    stats["last_session"] = {
        "model_name": new_metrics["model_name"],
        "cost_saved": new_metrics["cost_saved"],
        "time_saved": new_metrics["time_saved"],
        "carbon_footprint": new_metrics["carbon_footprint"],
        "tokens_used": new_metrics["tokens_used"]
    }
    
    save_stats(stats)
    return stats

# ==========================================
# APP LAYOUT
# ==========================================
def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Total Analytics"])

    if page == "Home":
        render_home_page()
    elif page == "Total Analytics":
        render_analytics_page()

def render_home_page():
    st.title("Intelligent Model Router")
    st.markdown("ask anything, and we'll route it to the **best** model.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input")
        
        # File Uploader (Plus Button concept)
        uploaded_files = st.file_uploader(
            "Upload photos or documents", 
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'pdf', 'txt', 'docx']
        )
        
        user_prompt = st.text_area("What can I help with?", height=150)
        
        # Dynamic Button Visibility Logic
        process_btn = st.button("Route & Process", type="primary")

        # METRICS AREA
        st.markdown("### Metrics (Current Session)")
        
        # Load initial state from JSON
        stats = load_stats()
        current_metrics = stats.get("last_session", {})

        # Logic Execution
        if process_btn:
            if not user_prompt and not uploaded_files:
                st.warning("Please enter a prompt or upload a file to proceed.")
            else:
                with st.spinner("Analyzing complexity and routing to optimal model..."):
                    # Call the backend logic
                    result = logic.get_routing_results(user_prompt, uploaded_files)
                    
                    # Update Persistent State in JSON
                    stats = update_stats(result)
                    current_metrics = stats["last_session"]
                    
                    # Save answer to session state to display in col2
                    st.session_state['last_answer'] = result['answer']
                    
                    # Success Feedback with Currency Conversion
                    cost_inr = result['cost_saved'] * 86.0
                    st.success(f"Successfully routed to {result['model_name']}! You saved ₹{cost_inr:.2f}")
                    
                    # Visual Reward for High Savings
                    if result['cost_saved'] > 0.4:
                         st.balloons()

        # Display Metrics in Container (Adjustable Height based on content)
        with st.container(border=True):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                cost_inr_display = current_metrics.get('cost_saved', 0) * 86.0
                st.metric("Model Used", current_metrics.get("model_name", "N/A"))
                st.metric("Cost Saved", f"₹{cost_inr_display:.2f}")
                st.metric("Carbon Footprint", f"{current_metrics.get('carbon_footprint', 0):.5f} kgCO2e")
            with m_col2:
                st.metric("Time Saved", f"{current_metrics.get('time_saved', 0):.2f} s")
                st.metric("Tokens Used", current_metrics.get("tokens_used", 0))

    with col2:
        st.subheader("Answer")
        # Professional Chat UI for the Response
        with st.chat_message("assistant"):
            if 'last_answer' in st.session_state:
                st.write(st.session_state['last_answer'])
            else:
                st.write("Awaiting prompt... Your AI-powered answer will appear here.")

def render_analytics_page():
    st.title("Total Analytics")
    st.markdown("### Lifetime Savings & Usage")
    
    stats = load_stats()
    
    # Grid of Lifetime Metrics
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            total_inr = stats.get('total_cost_saved', 0) * 86.0
            st.metric("Total Cost Saved", f"₹{total_inr:,.2f}")
            st.metric("Total Carbon Saved", f"{stats.get('total_carbon_saved', 0):.4f} kgCO2e")
        with c2:
            st.metric("Total Time Saved", f"{stats.get('total_time_saved', 0):.2f} s")
            st.metric("Total Tokens Saved", stats.get('total_tokens_saved', 0))

if __name__ == "__main__":
    main()