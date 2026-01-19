"""
Contextual RAG Document Upload Module
Phase 1: Cloud APIs (Claude/Gemini)
Phase 2: Local Ollama (Fully Integrated)
"""

import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Try to import from new location first, fall back to old location to prevent crashes
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

import os
import shutil
from typing import Dict, List, Optional

# IMPORT THE CENTRAL BRAIN
from model_config import get_llm

def intelligent_chunking(text):
    """
    Splits text by Financial Report headers first, then by size.
    """
    # 1. Split by logical document sections first
    headers_to_split_on = [
        ("##", "Section"),
        ("###", "Subsection"),
        ("Balance Sheet", "Financial_Table"),
        ("Board of Directors", "Entity_List")
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = markdown_splitter.split_text(text)

    # 2. Then ensure chunks fit in Llama 3's context window
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    return text_splitter.split_documents(header_splits)

def extract_pdf_text(pdf_file) -> str:
    """Extract text from PDF file with page markers"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text.strip():
                text += f"\n\n[Page {page_num + 1}]\n{page_text}"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def generate_pdf_summary(pdf_text: str, api_provider: str, api_key: str, model: str) -> Optional[str]:
    """Generate a global summary using the Central Brain"""
    try:
        # USE THE CENTRAL BRAIN (Handles Ollama or Cloud automatically)
        use_local = (api_provider == "Local/Ollama")
        # === FIX: PASS THE API KEY FROM SIDEBAR ===
        llm = get_llm(use_local=use_local, local_model="llama3", api_key=api_key)

        text_for_summary = pdf_text[:50000]
        prompt = f"""You are analyzing a document for a financial/business context.
Generate a concise 2-3 sentence summary.

DOCUMENT TEXT:
{text_for_summary}

SUMMARY:"""

        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return "Document containing business/financial information"

def generate_chunk_context(chunk: str, pdf_summary: str, api_provider: str, api_key: str, model: str) -> Optional[str]:
    """Generate context for a chunk using the Central Brain"""
    try:
        use_local = (api_provider == "Local/Ollama")
        # === FIX: PASS THE API KEY FROM SIDEBAR ===
        llm = get_llm(use_local=use_local, local_model="llama3", api_key=api_key)

        chunk_for_context = chunk[:5000]
        prompt = f"""Generate a brief 1-sentence "Cheat Note" for this chunk.

CONTEXT: {pdf_summary}
CHUNK: {chunk_for_context}

CHEAT NOTE:"""

        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Contains: {chunk[:100]}..."

def create_rag_vector_store(
    text: str,
    store_name: str,
    api_provider: str,
    api_key: str,
    model: str,
    progress_callback=None
) -> bool:
    """
    Create a vector store using Contextual RAG approach.
    """
    try:
        # Initialize embeddings (Local CPU)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Step 1: Generate global PDF summary
        if progress_callback:
            progress_callback(15, "📝 Generating document summary...")

        pdf_summary = generate_pdf_summary(text, api_provider, api_key, model)

        # Step 2: Split text into chunks using Intelligent Chunking
        if progress_callback:
            progress_callback(30, "✂️ Splitting document into chunks...")

        # === THE CUTTER IS HERE ===
        chunks_docs = intelligent_chunking(text)
        chunks = [doc.page_content for doc in chunks_docs]
        # ==========================

        # Step 3: Generate context
        if progress_callback:
            progress_callback(40, f"🧠 Generating context for {len(chunks)} chunks...")

        texts_with_context = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            chunk_context = generate_chunk_context(chunk, pdf_summary, api_provider, api_key, model)
            combined_text = f"Context: {chunk_context}\n\nContent: {chunk}"
            texts_with_context.append(combined_text)
            metadatas.append({
                "original_chunk": chunk,
                "chunk_context": chunk_context,
                "pdf_summary": pdf_summary,
                "chunk_index": idx,
                "chunk_count": len(chunks)
            })

            if progress_callback and idx % 5 == 0:
                progress = 40 + int((idx / len(chunks)) * 40)
                progress_callback(progress, f"🧠 Processing chunk {idx + 1}/{len(chunks)}...")

        # Step 4: Create vector store
        if progress_callback:
            progress_callback(80, "🔢 Creating vector embeddings...")

        vector_store = FAISS.from_texts(
            texts=texts_with_context,
            embedding=embeddings,
            metadatas=metadatas
        )

        # Step 5: Save
        if progress_callback:
            progress_callback(90, "💾 Saving vector store...")

        vector_store.save_local(f"comparison_{store_name}")

        with open(f"comparison_{store_name}_text.txt", "w", encoding="utf-8") as f:
            f.write(text)
        with open(f"comparison_{store_name}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"DOCUMENT SUMMARY:\n{pdf_summary}\n\nCHUNK COUNT: {len(chunks)}\n")

        if progress_callback:
            progress_callback(100, "✅ Vector store created successfully!")

        return True

    except Exception as e:
        print(f"Error in create_rag_vector_store: {e}")
        return False


def upload_and_process_rag(api_provider: str, api_key: str):
    """
    Main entry point for Contextual RAG document upload
    Called from app_test.py with API credentials
    """

    # Initialize session state
    if 'rag_doc1_processed' not in st.session_state:
        st.session_state.rag_doc1_processed = False
    if 'rag_doc2_processed' not in st.session_state:
        st.session_state.rag_doc2_processed = False

    # Header
    st.title("📤 Contextual RAG Document Upload")
    st.markdown("### Upload & Process Documents with Contextual RAG")

    st.markdown("""
    **What is Contextual RAG?**
    - 📝 Generates a global summary of your document
    - 🧠 Creates "Cheat Notes" for each section
    - 🔍 Embeds context + content for smarter retrieval
    - 📊 Preserves original content for accurate display
    """)

    st.info(
        "💡 **This program uses Contextual RAG for intelligent document processing.**\n"
        f"Using {api_provider} API for LLM processing."
    )

    # Check if documents already exist
    doc1_exists = os.path.exists("comparison_doc1")
    doc2_exists = os.path.exists("comparison_doc2")

    if doc1_exists and doc2_exists:
        st.success("✅ Both documents are already processed with Contextual RAG!")

        # Show document info
        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists("comparison_doc1_text.txt"):
                with open("comparison_doc1_text.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                    st.info(f"📄 **Document A**: {len(text):,} characters")
            if os.path.exists("comparison_doc1_summary.txt"):
                with open("comparison_doc1_summary.txt", "r", encoding="utf-8") as f:
                    summary = f.read()
                    with st.expander("📋 View Document A Summary"):
                        st.markdown(summary)

        with col2:
            if os.path.exists("comparison_doc2_text.txt"):
                with open("comparison_doc2_text.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                    st.info(f"📄 **Document B**: {len(text):,} characters")
            if os.path.exists("comparison_doc2_summary.txt"):
                with open("comparison_doc2_summary.txt", "r", encoding="utf-8") as f:
                    summary = f.read()
                    with st.expander("📋 View Document B Summary"):
                        st.markdown(summary)

        st.markdown("---")
        st.warning("⚠️ Upload new documents will overwrite existing ones!")

    # Document upload section
    st.markdown("---")
    st.subheader("📄 Upload Documents")

    col1, col2 = st.columns(2)

    # Document A
    with col1:
        st.markdown("#### 📄 Document A (Original/Reference)")
        doc1_file = st.file_uploader(
            "Upload Document A",
            type=['pdf'],
            key="doc1_upload_rag",
            help="Upload the first document"
        )

        if doc1_file:
            if st.button("🚀 Process with Contextual RAG", key="process_doc1_rag"):
                # Progress container
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(value, message):
                    progress_bar.progress(value / 100)
                    status_text.text(message)

                try:
                    # Step 1: Extract text
                    update_progress(5, "📄 Step 1/4: Extracting text from PDF...")
                    text = extract_pdf_text(doc1_file)

                    if text:
                        # Step 2-5: Create Contextual RAG vector store
                        update_progress(10, "📄 Step 2/4: Processing with Contextual RAG...")

                        if create_rag_vector_store(
                            text,
                            "doc1",
                            api_provider,
                            api_key,
                            "default",
                            update_progress
                        ):
                            st.session_state.rag_doc1_processed = True
                            st.success(f"✅ Document A processed successfully with Contextual RAG! ({len(text):,} characters)")
                            st.balloons()
                            st.rerun()
                        else:
                            status_text.error("❌ Error creating RAG vector store")
                    else:
                        status_text.error("❌ Could not extract text from PDF")
                except Exception as e:
                    status_text.error(f"❌ Error: {str(e)}")

        # Show status if already processed
        if os.path.exists("comparison_doc1"):
            st.success("✅ Document A ready (Contextual RAG)")
            if os.path.exists("comparison_doc1_text.txt"):
                with open("comparison_doc1_text.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                    st.caption(f"📊 Length: {len(text):,} characters")

    # Document B
    with col2:
        st.markdown("#### 📄 Document B (New/Modified)")
        doc2_file = st.file_uploader(
            "Upload Document B",
            type=['pdf'],
            key="doc2_upload_rag",
            help="Upload the second document"
        )

        if doc2_file:
            if st.button("🚀 Process with Contextual RAG", key="process_doc2_rag"):
                # Progress container
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(value, message):
                    progress_bar.progress(value / 100)
                    status_text.text(message)

                try:
                    # Step 1: Extract text
                    update_progress(5, "📄 Step 1/4: Extracting text from PDF...")
                    text = extract_pdf_text(doc2_file)

                    if text:
                        # Step 2-5: Create Contextual RAG vector store
                        update_progress(10, "📄 Step 2/4: Processing with Contextual RAG...")

                        if create_rag_vector_store(
                            text,
                            "doc2",
                            api_provider,
                            api_key,
                            "default",
                            update_progress
                        ):
                            st.session_state.rag_doc2_processed = True
                            st.success(f"✅ Document B processed successfully with Contextual RAG! ({len(text):,} characters)")
                            st.balloons()
                            st.rerun()
                        else:
                            status_text.error("❌ Error creating RAG vector store")
                    else:
                        status_text.error("❌ Could not extract text from PDF")
                except Exception as e:
                    status_text.error(f"❌ Error: {str(e)}")

        # Show status if already processed
        if os.path.exists("comparison_doc2"):
            st.success("✅ Document B ready (Contextual RAG)")
            if os.path.exists("comparison_doc2_text.txt"):
                with open("comparison_doc2_text.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                    st.caption(f"📊 Length: {len(text):,} characters")

    # Next step message
    if os.path.exists("comparison_doc1") and os.path.exists("comparison_doc2"):
        st.markdown("---")
        st.success("🎉 Both documents are ready with Contextual RAG!")

        st.info("""
        ✅ **Upload Complete!**

        Your documents have been processed using Contextual RAG:
        - ✅ Global summaries generated
        - ✅ Chunk contexts created
        - ✅ Smart embeddings built
        - ✅ Original content preserved

        Next step: Go back to the main menu and select **"Document Comparison"** to analyze differences.
        """)

    # Clear button
    st.markdown("---")
    if st.button("🗑️ Clear All Processed Documents"):
        try:
            # Clear RAG vector stores
            for store in ['comparison_doc1', 'comparison_doc2']:
                if os.path.exists(store):
                    shutil.rmtree(store)

            # Clear text files and summaries
            for file in ['comparison_doc1_text.txt', 'comparison_doc2_text.txt',
                         'comparison_doc1_summary.txt', 'comparison_doc2_summary.txt']:
                if os.path.exists(file):
                    os.remove(file)

            # Clear session state
            st.session_state.rag_doc1_processed = False
            st.session_state.rag_doc2_processed = False

            st.success("✅ All documents cleared!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Contextual RAG Upload",
        page_icon="📤",
        layout="wide"
    )
    # This should be called from app_test.py with API credentials
    st.error("This module should be called from app_test.py with API credentials")
