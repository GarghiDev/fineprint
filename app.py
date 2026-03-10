import streamlit as st
import os
import sys
from pathlib import Path

# Add the project root to Python path for Streamlit Cloud compatibility
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from retrieval.processor import DocumentProcessor
from retrieval.hybrid import HybridRetriever
from agents.workflow import PrivacyPolicyWorkflow

# Page config
st.set_page_config(
    page_title="FinePrint - Privacy Policy Analyzer", page_icon="🔍", layout="wide"
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    .verified-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .unverified-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .confidence-score {
        font-size: 1.1rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .citation {
        background-color: #1e1e2e;
        color: #c8b6ff;
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #9d7fd8;
        border-radius: 3px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False
    st.session_state.document_indices = {}
    st.session_state.chat_history = []


def load_documents():
    """Load and process all privacy policy documents."""
    with st.spinner(
        "Loading and processing privacy policies... This may take a minute."
    ):
        data_dir = "data"
        documents = {
            "TikTok": os.path.join(data_dir, "TikTok_Policy.txt"),
            "Meta": os.path.join(data_dir, "Meta_Policy.txt"),
            "Spotify": os.path.join(data_dir, "Spotify_Policy.txt"),
        }

        processor = DocumentProcessor(chunk_size=500, chunk_overlap=50)

        for doc_name, doc_path in documents.items():
            if os.path.exists(doc_path):
                chunks, faiss_index, bm25_index, embeddings = (
                    processor.process_document(doc_path)
                )

                retriever = HybridRetriever(chunks, faiss_index, bm25_index, embeddings)

                st.session_state.document_indices[doc_name] = {
                    "retriever": retriever,
                    "chunks": chunks,
                }
            else:
                st.error(f"Document not found: {doc_path}")

        st.session_state.documents_loaded = True


def display_result(result):
    """Display the answer with verification status and sources."""
    # Answer section
    st.markdown("### Answer")
    st.write(result["answer"])

    # Verification status
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        if result["verified"]:
            st.markdown(
                '<span class="verified-badge">✓ VERIFIED</span>', unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="unverified-badge">⚠ UNVERIFIED</span>',
                unsafe_allow_html=True,
            )

    with col2:
        confidence = result["confidence"]
        st.markdown(
            f'<div class="confidence-score">Confidence: {confidence}%</div>',
            unsafe_allow_html=True,
        )

    # Verification details
    if not result["verified"] and result["verification_issues"]:
        with st.expander("⚠️ Verification Issues", expanded=True):
            st.markdown("**Issues Found:**")
            for issue in result["verification_issues"]:
                st.markdown(f"- {issue}")

            if result["verification_feedback"]:
                st.markdown("**Feedback:**")
                st.write(result["verification_feedback"])

    # Retry information
    if result["retry_count"] > 0:
        st.info(
            f"Answer was refined {result['retry_count']} time(s) to improve accuracy."
        )

    # Sources section
    with st.expander("📚 View Sources", expanded=False):
        st.markdown(f"**Retrieved from {result['document']} Privacy Policy**")

        for i, source in enumerate(result["sources"]):
            chunk_idx = source["index"]
            score = source["score"]
            chunk_text = source["chunk"]

            is_cited = chunk_idx in result["citations"]

            if is_cited:
                st.markdown(
                    f"**[Chunk #{chunk_idx}]** ⭐ *Cited in answer* | Relevance: {score:.3f}"
                )
            else:
                st.markdown(f"**[Chunk #{chunk_idx}]** | Relevance: {score:.3f}")

            st.markdown(
                f'<div class="citation">{chunk_text[:500]}...</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")


# Main app
def main():
    st.markdown('<div class="main-header">🔍 FinePrint</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; font-size: 1.2rem; color: #666;">Multi-Agent Privacy Policy Analyzer with Hallucination Detection</p>',
        unsafe_allow_html=True,
    )

    # Load documents on first run (before sidebar)
    if not st.session_state.documents_loaded:
        load_documents()
        st.success("✅ All privacy policies loaded successfully!")
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Document selector
        selected_doc = st.selectbox(
            "Select Privacy Policy",
            options=list(st.session_state.document_indices.keys()),
            help="Choose which privacy policy to query",
        )

        st.markdown("---")

        # Information
        st.markdown("### About")
        st.markdown("""
        This tool uses a multi-agent RAG system to:
        - **Retrieve** relevant sections using hybrid BM25 + vector search
        - **Research** answers using Gemini AI
        - **Verify** responses to detect hallucinations
        - **Self-correct** with up to 2 retry attempts
        """)

        st.markdown("---")

        if st.button("🔄 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main chat interface
    st.markdown("---")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.write(message["content"])
            else:
                display_result(message["result"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the privacy policy..."):
        if not selected_doc:
            st.error("Please wait for documents to load.")
            return

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Analyzing privacy policy..."):
                # Get retriever for selected document
                retriever = st.session_state.document_indices[selected_doc]["retriever"]

                # Create workflow
                workflow = PrivacyPolicyWorkflow(
                    retriever=retriever, document_name=selected_doc, max_retries=2
                )

                # Run workflow
                result = workflow.run(prompt)

                # Display result
                display_result(result)

                # Add to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "result": result}
                )


if __name__ == "__main__":
    main()
