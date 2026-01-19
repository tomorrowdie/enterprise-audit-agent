"""
RAG Semantic Audit & Comparison (Unified App)
"""
import streamlit as st
import os
import shutil

# --- CONFIGURATION ---
st.set_page_config(page_title="RAG Semantic Audit", page_icon="🔬", layout="wide")

# --- IMPORTS ---
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from document_upload_rag import upload_and_process_rag
    from document_comparison_rag import run_semantic_audit, display_audit_results
except ImportError as e:
    st.error(f"CRITICAL ERROR: Missing Modules. {e}")
    st.stop()

def main():
    st.title("🔬 RAG Semantic Audit & Comparison")

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.title("⚙️ Configuration")

    # 1. Provider Selection
    # === FIX: Added "Local/Ollama" for Phase 2 ===
    api_provider = st.sidebar.radio("AI Provider:", ["Gemini", "Claude", "Local/Ollama"])

    # 2. API Key Logic
    if api_provider != "Local/Ollama":
        env_var = f"{api_provider.upper()}_API_KEY"
        env_key = os.getenv(env_var, "")
        if "your_" in env_key: env_key = ""

        placeholder_text = "Key loaded from .env" if env_key else "Enter API Key"
        api_key_input = st.sidebar.text_input(f"{api_provider} API Key:", type="password", placeholder=placeholder_text)
        final_api_key = api_key_input if api_key_input else env_key
    else:
        # Local mode needs no key
        final_api_key = "dummy_local_key"
        st.sidebar.info("[LOCAL] Using Local Ollama (No Key Required)")

    # 3. Model Selector
    if api_provider == "Gemini":
        model_options = ["gemini-2.5-flash", "gemini-2.5-pro"]
    elif api_provider == "Claude":
        model_options = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]
    else:
        # === FIX: Options for Ollama ===
        model_options = ["llama3", "mistral", "gemma2"]

    selected_model = st.sidebar.selectbox("Model:", model_options)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📤 Upload Documents", "🔍 Vector Compare", "🔬 Semantic Audit"])

    # --- TAB 1: UPLOAD ---
    with tab1:
        if final_api_key:
            upload_and_process_rag(api_provider, final_api_key)
        else:
            st.warning(f"⚠️ Please enter your {api_provider} API Key in the sidebar.")

    # --- TAB 2: VECTOR COMPARE ---
    with tab2:
        if os.path.exists("comparison_doc1") and os.path.exists("comparison_doc2"):
            st.success("✅ Vector Stores Ready")
            st.info("Visual comparison feature coming soon.")
        else:
            st.info("ℹ️ Please upload documents in Tab 1 first.")

    # --- TAB 3: SEMANTIC AUDIT ---
    with tab3:
        st.subheader("Semantic Variance Analysis")

        # Check for files
        doc1_ready = os.path.exists("comparison_doc1_text.txt")
        doc2_ready = os.path.exists("comparison_doc2_text.txt")

        if doc1_ready and doc2_ready:
            st.success("✅ Using Your Uploaded Documents")

            # Audit Mode Selector
            audit_mode = st.radio("Audit Mode:", ["Error Detection", "Variance Analysis (YoY)"])

            # Initialize State
            if 'audit_results' not in st.session_state:
                st.session_state['audit_results'] = None

            if st.button("🚀 Run Audit", type="primary"):
                if not final_api_key:
                    st.error("❌ Missing API Key.")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(val, msg):
                        progress_bar.progress(val / 100)
                        status_text.text(msg)

                    # Run the Audit
                    results = run_semantic_audit(
                        "comparison_doc1_text.txt", # Source Path
                        "comparison_doc2_text.txt", # Target Path
                        api_provider,
                        final_api_key,
                        selected_model,
                        update_progress,
                        audit_mode=audit_mode
                    )

                    st.session_state['audit_results'] = results
                    status_text.empty()
                    progress_bar.empty()

            # Show Results
            if st.session_state['audit_results']:
                display_audit_results(st.session_state['audit_results'])

        else:
            st.warning("⚠️ No documents found. Please upload files in Tab 1.")

if __name__ == "__main__":
    main()
