import streamlit as st
import logic
import os
import json
from dotenv import load_dotenv
import base64
from PIL import Image
import re

load_dotenv()

# ==========================================
# CONFIGURATION & STYLE
# ==========================================
st.set_page_config(
    layout="wide", 
    page_title="Egnyte Copilot",
    initial_sidebar_state="expanded"
)

# Egnyte Colors
EGNYTE_BLUE = "#007BFF"
EGNYTE_GREEN = "#00C2B2"
EGNYTE_NAVY = "#112532" 

# Custom CSS
st.markdown(f"""
    <style>
    /* 1. RESET THEME */
    .stApp {{
        background-color: #ffffff;
    }}
    
    /* Hide top bar decoration but keep sidebar toggle clickable */
    header[data-testid="stHeader"] {{
        background-color: transparent;
    }}
    .block-container {{
        padding-top: 1rem !important; 
    }}
    #MainMenu {{
        visibility: hidden;
    }}
    footer {{
        visibility: hidden;
    }}
    
    /* 2. GLOBAL TEXT COLORS - Dark Navy */
    h1, h2, h3, h4, h5, h6, p, div, span, label, li, .stMarkdown {{
        color: {EGNYTE_NAVY} !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* 3. SIDEBAR SPECIFIC OVERRIDES */
    [data-testid="stSidebar"] {{
        background-color: #14264A !important;
    }}
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] input {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] .stRadio label {{
        color: #ffffff !important;
    }}
    
    /* 4. FILE UPLOADER - LIGHT THEME */
    [data-testid='stFileUploader'] {{
        background-color: #ffffff;
    }}
    [data-testid='stFileUploader'] section {{
        background-color: #f7f9fc !important;
        border: 1px dashed #ced4da !important;
    }}
    [data-testid='stFileUploader'] button {{
        background-color: white !important;
        color: {EGNYTE_NAVY} !important;
        border: 1px solid #ced4da !important;
    }}

    /* 5. TITLE STYLING */
    .egnyte-title {{
        color: {EGNYTE_GREEN} !important;
        font-weight: bold;
        font-size: 3rem;
        vertical-align: middle;
    }}

    /* 6. METRICS */
    [data-testid="stMetricValue"] {{
        color: {EGNYTE_GREEN} !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: #5D636E !important; 
    }}
    div[data-testid="stMetric"] {{
        background-color: #f7f9fc !important;
        border: 1px solid #e2e5eb !important;
    }}
    
    /* 7. PRIMARY BUTTON */
    div.stButton > button:first-child {{
        background-color: {EGNYTE_GREEN} !important;
        color: white !important;
        border: none;
        font-weight: bold;
    }}
    div.stButton > button:hover {{
        background-color: #009e91 !important;
        color: white !important;
        border: none;
    }}
    
    /* 8. INFO BOX */
    .stAlert {{
        background-color: #f0fbfb !important;
        border-left-color: {EGNYTE_GREEN} !important;
        color: {EGNYTE_NAVY} !important;
    }}
    
    /* 9. INPUT AREA */
    .stTextArea textarea {{
        background-color: #ffffff !important;
        color: {EGNYTE_NAVY} !important;
        border: 1px solid #ced4da !important;
        caret-color: #000000 !important;
    }}
    
    /* 10. STREAMING CURSOR */
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0; }}
    }}
    .streaming-cursor {{
        display: inline-block;
        width: 8px;
        height: 16px;
        background-color: {EGNYTE_GREEN} !important;
        margin-left: 2px;
        animation: blink 1s infinite;
        vertical-align: middle;
    }}
    
    /* 11. CUSTOM CODE BLOCK STYLING */
    .custom-code-block {{
        background-color: #f7f9fc;
        border: 1px solid #e2e5eb;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        font-family: 'Courier New', monospace;
        position: relative;
    }}
    
    .custom-code-block .code-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e2e5eb;
    }}
    
    .custom-code-block .code-language {{
        color: {EGNYTE_GREEN};
        font-weight: bold;
        font-size: 0.85rem;
        text-transform: uppercase;
    }}
    
    .custom-code-block .code-copy-btn {{
        background-color: #ffffff;
        color: {EGNYTE_NAVY};
        border: 1px solid #e2e5eb;
        padding: 4px 12px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 0.85rem;
        transition: all 0.2s;
    }}
    
    .custom-code-block .code-copy-btn:hover {{
        background-color: #e2e5eb;
    }}
    
    .custom-code-block pre {{
        margin: 0;
        padding: 0;
        background-color: transparent;
        overflow-x: auto;
    }}
    
    .custom-code-block code {{
        color: {EGNYTE_NAVY};
        background-color: transparent;
        font-size: 0.95rem;
        line-height: 1.6;
        display: block;
        white-space: pre;
    }}
    </style>
    """, unsafe_allow_html=True)

STATS_FILE = "stats.json"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def load_stats():
    default_stats = {
        "total_time_saved": 0.0,
        "total_cost_saved": 0.0,
        "total_carbon_saved": 0.0,
        "last_session": {
            "model_name": "N/A",
            "cost_saved": 0.0,
            "time_saved": 0.0,
            "carbon_footprint": 0.0
        }
    }
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r") as f:
                data = json.load(f)
                return data
        except Exception:
            return default_stats
    return default_stats

def save_stats(stats):
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=4)

def update_stats(new_metrics):
    stats = load_stats()
    stats["total_time_saved"] += new_metrics["time_saved"]
    stats["total_cost_saved"] += new_metrics["cost_saved"]
    stats["total_carbon_saved"] += new_metrics["carbon_footprint"]
    stats["last_session"] = {
        "model_name": new_metrics["model_name"],
        "cost_saved": new_metrics["cost_saved"],
        "time_saved": new_metrics["time_saved"],
        "carbon_footprint": new_metrics["carbon_footprint"]
    }
    save_stats(stats)
    return stats

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def format_code_block(text):
    """
    Convert markdown code blocks to custom HTML with proper styling.
    Handles both with and without language specification.
    """
    if "```" not in text:
        return text
    
    # Pattern to match code blocks with optional language
    pattern = r'```(\w+)?\n(.*?)```'
    
    def replace_code(match):
        language = match.group(1) if match.group(1) else "text"
        code_content = match.group(2)
        
        # Escape HTML to prevent rendering issues
        code_content = code_content.replace('<', '&lt;').replace('>', '&gt;')
        
        # Generate unique ID for copy functionality
        code_id = f"code_{hash(code_content) % 10000}"
        
        return f'''
<div class="custom-code-block">
    <div class="code-header">
        <span class="code-language">{language}</span>
        <button class="code-copy-btn" onclick="navigator.clipboard.writeText(document.getElementById('{code_id}').innerText)">
            Copy
        </button>
    </div>
    <pre><code id="{code_id}">{code_content}</code></pre>
</div>
'''
    
    # Replace all code blocks
    formatted_text = re.sub(pattern, replace_code, text, flags=re.DOTALL)
    
    return formatted_text

# ==========================================
# APP LAYOUT
# ==========================================
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Total Analytics"])

    if page == "Home":
        render_home_page()
    elif page == "Total Analytics":
        render_analytics_page()

def render_home_page():
    # BRANDED HEADER
    c1, c2, c3 = st.columns([1, 6, 1])
    
    with c2:
        if os.path.exists("egnyte_logo.jpg"):
            img_base64 = get_base64_image("egnyte_logo.jpg")
            img_src = f"data:image/jpg;base64,{img_base64}"
        else:
            img_src = "" 

        st.markdown(f"""
            <div style="text-align: center;">
                <img src="{img_src}" width="50" style="vertical-align: middle; margin-right: 15px;">
                <span class="egnyte-title">Egnyte Copilot</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"<div style='text-align: center; margin-bottom: 30px; color: {EGNYTE_NAVY};'>ask anything, and we'll route it to the <b>best</b> model.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded_files = st.file_uploader(
            "Upload photos or documents", 
            accept_multiple_files=True,
            type=['png', 'jpg', 'jpeg', 'pdf', 'txt', 'docx']
        )
        
        user_prompt = st.text_area("What can I help with?", height=150)
        
        process_btn = st.button("Route & Process", type="primary")
        status_placeholder = st.empty()

        st.markdown("### Metrics (Current Session)")
        
        stats = load_stats()
        current_metrics = stats.get("last_session", {})
        metrics_placeholder = st.empty()
        
        def render_metrics(metrics):
            with metrics_placeholder.container():
                with st.container(border=True):
                    m1, m2 = st.columns(2)
                    with m1:
                        cost_val = metrics.get('cost_saved', 0) * 86.0
                        st.metric("Model Used", metrics.get("model_name", "N/A"))
                        st.metric("Cost Saved", f"₹{cost_val:.2f}")
                    with m2:
                        st.metric("Time Saved", f"{metrics.get('time_saved', 0):.2f} s")
                        st.metric("Carbon Footprint", f"{metrics.get('carbon_footprint', 0):.5f} kgCO2e")
        
        render_metrics(current_metrics)

    with col2:
        st.subheader("Answer")
        
        try:
            icon_img = Image.open("egnyte_logo1.jpg")
        except:
            icon_img = "🤖"

        answer_container = st.chat_message("assistant", avatar=icon_img)
        
        if not process_btn:
            if 'last_answer' in st.session_state:
                # Format the saved answer with custom code blocks
                formatted_answer = format_code_block(st.session_state['last_answer'])
                answer_container.markdown(formatted_answer, unsafe_allow_html=True)
            else:
                answer_container.write("Awaiting prompt... Your AI-powered answer will appear here.")

    # ==========================================
    # STREAMING LOGIC
    # ==========================================
    if process_btn:
        if not user_prompt and not uploaded_files:
            st.warning("Please enter a prompt or upload a file to proceed.")
        else:
            if 'streaming_complete' not in st.session_state:
                st.session_state['streaming_complete'] = False
            
            status_placeholder.markdown("🔄 **Analyzing complexity and routing...**")
            
            stream_generator = logic.get_routing_results_stream(user_prompt, uploaded_files)
            
            try:
                first_chunk = next(stream_generator)
            except StopIteration:
                status_placeholder.error("Router failed to respond.")
                return
            
            if isinstance(first_chunk, dict):
                metadata = first_chunk
                chosen_model = metadata['model_name']
                reason = metadata.get('routing_reason', 'Heuristic routing.')
                
                render_metrics(metadata)
                
                cost_inr = metadata['cost_saved'] * 86.0
                status_placeholder.markdown(f"✅ **Routed to {chosen_model}!** Estimated savings: **₹{cost_inr:.2f}**")
                
                with answer_container:
                     st.info(f"**Why this model?**\n\n{reason}", icon="🧠")
                
                st.session_state['current_model'] = chosen_model
                st.session_state['final_metrics'] = metadata
                
                with answer_container:
                    text_placeholder = st.empty()
                    full_response = ""
                    
                    for chunk in stream_generator:
                        if isinstance(chunk, str):
                            full_response += chunk
                            
                            # Format code blocks during streaming
                            formatted_response = format_code_block(full_response)
                            
                            text_placeholder.markdown(
                                formatted_response + '<span class="streaming-cursor"></span>',
                                unsafe_allow_html=True
                            )
                    
                    # Final render without cursor
                    formatted_final = format_code_block(full_response)
                    text_placeholder.markdown(formatted_final, unsafe_allow_html=True)
                    
                    # Save the raw response (without formatting) for future reference
                    st.session_state['last_answer'] = full_response
                
                stats = update_stats(metadata)
                current_metrics = stats["last_session"]
                
                if metadata['cost_saved'] > 0.4:
                    st.balloons()

def render_analytics_page():
    st.title("Total Analytics")
    st.markdown("### Lifetime Savings & Usage")
    
    stats = load_stats()
    
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            total_inr = stats.get('total_cost_saved', 0) * 86.0
            st.metric("Total Cost Saved", f"₹{total_inr:,.2f}")
        with c2:
            st.metric("Total Time Saved", f"{stats.get('total_time_saved', 0):.2f} s")
        with c3:
            st.metric("Total Carbon Saved", f"{stats.get('total_carbon_saved', 0):.4f} kgCO2e")

if __name__ == "__main__":
    main()