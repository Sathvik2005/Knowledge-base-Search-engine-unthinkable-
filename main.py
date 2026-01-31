"""
Answer IQ (RAG-Gemini) - Main Streamlit Application
====================================================
A professional Retrieval-Augmented Generation system with voice support.
Designed for privacy-preserving, offline document question-answering.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Disable TensorFlow to avoid Keras 3 conflicts
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

import streamlit as st
from datetime import datetime
import json

# Import application modules
from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS,
    EMBEDDING_MODEL_NAME, GENERATION_MODEL_NAME, WHISPER_MODEL_SIZE,
    WHISPER_MODEL_NAME, HISTORY_FILE, CACHE_DIR
)
from src.ingestion import IngestionPipeline
from src.embeddings import EmbeddingModel
from src.retrieval import FAISSIndex, Retriever
from src.generation import AnswerGenerator, QueryProcessor
from src.audio import AudioProcessor


# =============================================================================
# Session State Initialization
# =============================================================================

def initialize_session_state():
    """Initialize all session state variables."""
    
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if 'ingestion_pipeline' not in st.session_state:
        st.session_state.ingestion_pipeline = IngestionPipeline(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
    
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = None
    
    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None
    
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = None
    
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = AudioProcessor(model_name=WHISPER_MODEL_NAME)
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = load_history()
    
    if 'kb_ready' not in st.session_state:
        st.session_state.kb_ready = False
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = []


def load_history():
    """Load query history from file."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_history(history):
    """Save query history to file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history[-100:], f, ensure_ascii=False, indent=2)


# =============================================================================
# Core Functions
# =============================================================================

def initialize_models():
    """Initialize embedding and generation models."""
    with st.spinner("Loading embedding model..."):
        st.session_state.embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
        st.session_state.embedding_model.load()
    
    with st.spinner("Initializing FAISS index..."):
        dimension = st.session_state.embedding_model.dimension
        st.session_state.faiss_index = FAISSIndex(dimension=dimension)
    
    with st.spinner("Loading generation model..."):
        st.session_state.generator = AnswerGenerator(GENERATION_MODEL_NAME)
        st.session_state.generator.load()
    
    # Create retriever
    st.session_state.retriever = Retriever(
        st.session_state.embedding_model,
        st.session_state.faiss_index
    )
    
    # Create query processor
    st.session_state.query_processor = QueryProcessor(
        st.session_state.retriever,
        st.session_state.generator
    )
    
    st.session_state.initialized = True


def process_documents(uploaded_files):
    """Process uploaded documents and build knowledge base."""
    if not uploaded_files:
        return False, "No files uploaded."
    
    # Ensure models are initialized
    if not st.session_state.initialized:
        initialize_models()
    
    # Process files
    pipeline = st.session_state.ingestion_pipeline
    doc_count, chunk_count, errors = pipeline.process_files(uploaded_files)
    
    if errors:
        for error in errors:
            st.session_state.processing_status.append(f"Error: {error}")
    
    if chunk_count == 0:
        return False, "No text content could be extracted from the uploaded files."
    
    # Index chunks
    with st.spinner(f"Indexing {chunk_count} text chunks..."):
        indexed = st.session_state.retriever.index_chunks(
            pipeline.all_chunks,
            show_progress=False
        )
    
    st.session_state.kb_ready = True
    
    return True, f"Successfully processed {doc_count} documents with {chunk_count} chunks."


def execute_query(query: str, concise: bool = True):
    """Execute a query through the RAG pipeline."""
    if not st.session_state.kb_ready:
        return None, "Knowledge base is not ready. Please upload and process documents first."
    
    if not query or not query.strip():
        return None, "Please enter a valid question."
    
    # Process query
    result = st.session_state.query_processor.process(
        query=query,
        top_k=TOP_K_RESULTS,
        concise=concise
    )
    
    # Save to history
    history_entry = {
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': result.answer,
        'sources': result.sources,
        'confidence': result.confidence
    }
    st.session_state.query_history.append(history_entry)
    save_history(st.session_state.query_history)
    
    return result, None


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with document upload and status."""
    
    st.sidebar.markdown("### 📚 Answer IQ")
    st.sidebar.caption("Enterprise RAG Knowledge Base")
    
    st.sidebar.markdown("---")
    
    # Document Upload Section
    st.sidebar.subheader("Document Upload")
    
    uploaded_files = st.sidebar.file_uploader(
        "Select PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload documents to build your knowledge base"
    )
    
    if uploaded_files:
        st.sidebar.info(f"{len(uploaded_files)} file(s) selected")
        
        if st.sidebar.button("Process Documents", type="primary", use_container_width=True):
            success, message = process_documents(uploaded_files)
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)
    
    st.sidebar.markdown("---")
    
    # Knowledge Base Status
    st.sidebar.subheader("Knowledge Base Status")
    
    if st.session_state.kb_ready:
        stats = st.session_state.ingestion_pipeline.get_statistics()
        
        col1, col2 = st.sidebar.columns(2)
        col1.metric("Documents", stats['document_count'])
        col2.metric("Chunks", stats['chunk_count'])
        
        st.sidebar.metric("Total Words", f"{stats['total_words']:,}")
        
        if stats['documents']:
            with st.sidebar.expander("Document Details"):
                for doc in stats['documents']:
                    st.text(f"{doc['filename']}")
                    st.caption(f"  Pages: {doc['pages']} | Words: {doc['words']:,}")
    else:
        st.sidebar.warning("No documents loaded")
    
    st.sidebar.markdown("---")
    
    # System Information
    with st.sidebar.expander("System Configuration"):
        st.text(f"Embedding: {EMBEDDING_MODEL_NAME.split('/')[-1]}")
        st.text(f"Generator: {GENERATION_MODEL_NAME.split('/')[-1]}")
        st.text(f"Whisper: {WHISPER_MODEL_SIZE}")
        st.text(f"Chunk Size: {CHUNK_SIZE}")
        st.text(f"Top-K: {TOP_K_RESULTS}")
    
    # Clear Button
    if st.sidebar.button("Clear Knowledge Base", use_container_width=True):
        st.session_state.ingestion_pipeline.clear()
        if st.session_state.faiss_index:
            st.session_state.faiss_index.clear()
        st.session_state.kb_ready = False
        st.sidebar.info("Knowledge base cleared")
        st.rerun()


def render_text_query_tab():
    """Render the text query interface."""
    
    st.subheader("📝 Text Query")
    st.caption("Enter your question below to search the knowledge base")
    
    # Query input
    query = st.text_area(
        "Your Question",
        height=100,
        placeholder="Type your question here...",
        help="Enter a natural language question about your documents"
    )
    
    # Options
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        concise_mode = st.checkbox("Concise Answer", value=True)
    
    with col2:
        submit_button = st.button("Submit Query", type="primary")
    
    # Process query
    if submit_button:
        if not st.session_state.kb_ready:
            st.error("Please upload and process documents first.")
        elif not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing query..."):
                result, error = execute_query(query, concise=concise_mode)
            
            if error:
                st.error(error)
            else:
                render_answer(result)


def render_voice_query_tab():
    """Render the voice query interface with live microphone recording."""
    
    st.subheader("🎤 Voice Query")
    
    # Browser permission notice - clean info box
    st.markdown('''
    <div class="info-panel">
        <p style="color: #E2E8F0; margin: 0 0 0.5rem 0; font-weight: 600;">🔊 Microphone Access Required</p>
        <p style="color: #94A3B8; margin: 0; font-size: 0.9rem; line-height: 1.6;">
            When you click the microphone button, your browser will request permission. 
            Click <strong>Allow</strong>, ensure your microphone is connected, and speak clearly.
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("")  # Spacing
    
    # Check if audio processing is available
    if not st.session_state.audio_processor.is_available():
        st.error("Voice processing is not available. Please install transformers library.")
        return
    
    # Create tabs for different input methods
    voice_tab1, voice_tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload Audio File"])
    
    with voice_tab1:
        st.markdown("##### Record Your Question")
        st.caption("Click the microphone button below to start recording")
        
        # Streamlit audio input for live recording
        audio_recording = st.audio_input(
            "Click to record your question",
            key="voice_recorder",
            help="Click the microphone icon to start recording. Click again to stop."
        )
        
        if audio_recording:
            st.success("Audio recorded successfully")
            st.audio(audio_recording, format="audio/wav")
            
            # Transcribe button
            if st.button("Transcribe & Search", type="primary", key="btn_rec"):
                if not st.session_state.kb_ready:
                    st.error("Please upload and process documents first.")
                else:
                    with st.spinner("Loading Whisper model and transcribing..."):
                        text, error = st.session_state.audio_processor.process_microphone_input(
                            audio_recording
                        )
                    
                    if error:
                        st.error(f"Transcription failed: {error}")
                    elif not text.strip():
                        st.warning("No speech detected in the recording.")
                    else:
                        st.session_state.transcribed_text = text
                        st.rerun()
        
        # Show transcription result if available
        if 'transcribed_text' in st.session_state and st.session_state.transcribed_text:
            st.markdown("---")
            st.markdown("##### Transcription Result")
            
            edited_text = st.text_area(
                "Edit transcribed text if needed:",
                value=st.session_state.transcribed_text,
                height=100,
                key="edited_transcription_rec"
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                concise_mode = st.checkbox("Concise Answer", value=True, key="voice_concise_rec")
            with col2:
                if st.button("Search Knowledge Base", type="primary", key="search_rec"):
                    with st.spinner("Processing query..."):
                        result, query_error = execute_query(edited_text, concise=concise_mode)
                    
                    if query_error:
                        st.error(query_error)
                    else:
                        render_answer(result)
                        st.session_state.transcribed_text = None
            
            if st.button("Clear", key="clear_rec"):
                st.session_state.transcribed_text = None
                st.rerun()
    
    with voice_tab2:
        st.markdown("##### Upload Audio File")
        st.caption("Upload a pre-recorded audio file (WAV, MP3, M4A, OGG, FLAC)")
        
        audio_file = st.file_uploader(
            "Select audio file",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            help="Supported formats: WAV, MP3, M4A, OGG, FLAC",
            key="voice_upload"
        )
        
        if audio_file:
            st.audio(audio_file)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                concise_mode_upload = st.checkbox("Concise Answer", value=True, key="voice_concise_upload")
            with col2:
                if st.button("Transcribe & Search", type="primary", key="btn_upload"):
                    if not st.session_state.kb_ready:
                        st.error("Please upload and process documents first.")
                    else:
                        with st.spinner("Transcribing audio file..."):
                            text, error = st.session_state.audio_processor.process_audio_query(
                                audio_file, source_type="upload"
                            )
                        
                        if error:
                            st.error(f"Transcription failed: {error}")
                        elif not text.strip():
                            st.warning("No speech detected in the audio file.")
                        else:
                            st.success("Transcription successful")
                            st.markdown(f'''
                            <div class="info-panel">
                                <p style="color: #94A3B8; font-size: 0.8rem; margin: 0 0 0.5rem 0;">TRANSCRIBED TEXT</p>
                                <p style="color: #E2E8F0; margin: 0;">{text}</p>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            with st.spinner("Processing query..."):
                                result, query_error = execute_query(text, concise=concise_mode_upload)
                            
                            if query_error:
                                st.error(query_error)
                            else:
                                render_answer(result)


def render_answer(result):
    """Render the answer section with professional styling."""
    
    st.markdown("---")
    st.subheader("💡 Answer")
    
    # Main answer with Navy + Royal Blue styling
    st.markdown(
        f'''<div class="answer-card">
        <p>{result.answer}</p>
        </div>''',
        unsafe_allow_html=True
    )
    
    # Metadata with clean metrics
    st.markdown('<p class="section-header">Response Details</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_pct = f"{result.confidence * 100:.1f}%"
        st.metric("Confidence", confidence_pct)
    
    with col2:
        st.metric("Sources Used", len(result.sources))
    
    with col3:
        if result.is_grounded:
            st.markdown('<span class="status-badge success">Grounded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge warning">Ungrounded</span>', unsafe_allow_html=True)
    
    # Sources
    if result.sources:
        with st.expander("Source Documents"):
            for source in result.sources:
                st.text(f"- {source}")
    
    # Context (optional)
    if result.context_used:
        with st.expander("Retrieved Context"):
            st.text(result.context_used[:2000] + "..." if len(result.context_used) > 2000 else result.context_used)


def render_history_tab():
    """Render the query history."""
    
    st.subheader("📋 Query History")
    
    history = st.session_state.query_history
    
    if not history:
        st.info("No query history yet. Start asking questions to build your history.")
        return
    
    # Show recent queries (reversed)
    for entry in reversed(history[-20:]):
        with st.expander(f"{entry['query'][:80]}..."):
            st.caption(f"Time: {entry['timestamp']}")
            st.markdown(f"**Question:** {entry['query']}")
            st.markdown(f"**Answer:** {entry['answer']}")
            if entry.get('sources'):
                st.caption(f"Sources: {', '.join(entry['sources'])}")
    
    # Clear history button
    if st.button("Clear History"):
        st.session_state.query_history = []
        save_history([])
        st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="Answer IQ - RAG Knowledge Base",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # ==========================================================================
    # Professional Navy + Royal Blue Theme CSS
    # ==========================================================================
    # Design System:
    # - Navy Background: #0F172A (deep), #1E293B (secondary)
    # - Royal Blue: #2563EB (primary), #1D4ED8 (hover), #3B82F6 (light)
    # - Text: #E2E8F0 (primary), #94A3B8 (secondary), #64748B (muted)
    # - Borders: #334155 (subtle)
    # - Success: #22C55E | Warning: #F59E0B | Error: #EF4444 | Info: #3B82F6
    # ==========================================================================
    
    st.markdown("""
    <style>
    /* ==========================================================================
       BASE STYLES - Typography & Layout
       ========================================================================== */
    
    /* Root variables for consistent theming */
    :root {
        --navy-900: #0F172A;
        --navy-800: #1E293B;
        --navy-700: #334155;
        --navy-600: #475569;
        --royal-blue: #2563EB;
        --royal-blue-hover: #1D4ED8;
        --royal-blue-light: #3B82F6;
        --text-primary: #E2E8F0;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --border-color: #334155;
        --success: #22C55E;
        --warning: #F59E0B;
        --error: #EF4444;
        --info: #3B82F6;
    }
    
    /* Main container padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headings - clean hierarchy without color dependency */
    h1 {
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.025em;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.02em;
    }
    
    h3 {
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    
    /* Body text */
    p, span, label {
        color: var(--text-secondary);
    }
    
    /* Captions - muted text */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
    }
    
    /* ==========================================================================
       SIDEBAR STYLES
       ========================================================================== */
    
    [data-testid="stSidebar"] {
        background-color: var(--navy-800) !important;
        border-right: 1px solid var(--border-color) !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
    
    /* Sidebar title styling */
    [data-testid="stSidebar"] .stMarkdown h1:first-of-type {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        border-bottom: 2px solid var(--royal-blue);
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    
    /* ==========================================================================
       BUTTON STYLES
       ========================================================================== */
    
    /* Primary buttons - Royal Blue */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: var(--royal-blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.25rem !important;
        transition: background-color 0.15s ease !important;
        box-shadow: none !important;
    }
    
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        background-color: var(--royal-blue-hover) !important;
        box-shadow: none !important;
    }
    
    /* Secondary buttons - outline style */
    .stButton > button[kind="secondary"],
    .stButton > button[data-testid="baseButton-secondary"] {
        background-color: transparent !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }
    
    .stButton > button[kind="secondary"]:hover,
    .stButton > button[data-testid="baseButton-secondary"]:hover {
        background-color: var(--navy-700) !important;
        border-color: var(--text-muted) !important;
        box-shadow: none !important;
    }
    
    /* Default buttons */
    .stButton > button {
        background-color: var(--navy-700) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease !important;
        box-shadow: none !important;
    }
    
    .stButton > button:hover {
        background-color: var(--navy-600) !important;
        box-shadow: none !important;
    }
    
    /* Remove focus glow */
    .stButton > button:focus {
        box-shadow: 0 0 0 2px var(--royal-blue) !important;
        outline: none !important;
    }
    
    /* ==========================================================================
       INPUT STYLES
       ========================================================================== */
    
    /* Text inputs, text areas, select boxes */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div {
        background-color: var(--navy-800) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        transition: border-color 0.15s ease !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--royal-blue) !important;
        box-shadow: 0 0 0 1px var(--royal-blue) !important;
        outline: none !important;
    }
    
    /* Placeholder text */
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* ==========================================================================
       TAB STYLES
       ========================================================================== */
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0 !important;
        background-color: var(--navy-800) !important;
        border-radius: 8px !important;
        padding: 4px !important;
        border: 1px solid var(--border-color) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 44px !important;
        padding: 0 24px !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        background-color: transparent !important;
        border-radius: 6px !important;
        border: none !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background-color: var(--navy-700) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--royal-blue) !important;
        color: white !important;
    }
    
    /* Remove tab highlight bar */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }
    
    /* ==========================================================================
       CARD & CONTAINER STYLES
       ========================================================================== */
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: var(--navy-800) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: var(--navy-700) !important;
    }
    
    .streamlit-expanderContent {
        background-color: var(--navy-800) !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }
    
    /* ==========================================================================
       METRIC STYLES
       ========================================================================== */
    
    [data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
    }
    
    /* ==========================================================================
       ALERT & STATUS STYLES
       ========================================================================== */
    
    /* Success - muted green */
    .stSuccess {
        background-color: rgba(34, 197, 94, 0.1) !important;
        border-left: 4px solid var(--success) !important;
        border-radius: 0 6px 6px 0 !important;
        color: var(--text-primary) !important;
    }
    
    /* Warning - soft amber */
    .stWarning {
        background-color: rgba(245, 158, 11, 0.1) !important;
        border-left: 4px solid var(--warning) !important;
        border-radius: 0 6px 6px 0 !important;
        color: var(--text-primary) !important;
    }
    
    /* Error - restrained red */
    .stError {
        background-color: rgba(239, 68, 68, 0.1) !important;
        border-left: 4px solid var(--error) !important;
        border-radius: 0 6px 6px 0 !important;
        color: var(--text-primary) !important;
    }
    
    /* Info - royal blue variant */
    .stInfo {
        background-color: rgba(59, 130, 246, 0.1) !important;
        border-left: 4px solid var(--info) !important;
        border-radius: 0 6px 6px 0 !important;
        color: var(--text-primary) !important;
    }
    
    /* ==========================================================================
       FILE UPLOADER STYLES
       ========================================================================== */
    
    [data-testid="stFileUploader"] {
        background-color: var(--navy-800) !important;
        border: 2px dashed var(--border-color) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--royal-blue) !important;
    }
    
    /* ==========================================================================
       AUDIO INPUT STYLES
       ========================================================================== */
    
    [data-testid="stAudioInput"] > div {
        background-color: var(--navy-800) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
    }
    
    /* ==========================================================================
       DIVIDER STYLES
       ========================================================================== */
    
    hr {
        border-color: var(--border-color) !important;
        opacity: 0.5 !important;
    }
    
    /* ==========================================================================
       CHECKBOX & RADIO STYLES
       ========================================================================== */
    
    .stCheckbox label span {
        color: var(--text-secondary) !important;
    }
    
    /* ==========================================================================
       SPINNER STYLES
       ========================================================================== */
    
    .stSpinner > div {
        border-top-color: var(--royal-blue) !important;
    }
    
    /* ==========================================================================
       LINK STYLES
       ========================================================================== */
    
    a {
        color: var(--royal-blue) !important;
        text-decoration: none !important;
    }
    
    a:hover {
        text-decoration: underline !important;
    }
    
    /* ==========================================================================
       SCROLLBAR STYLES
       ========================================================================== */
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--navy-800);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--navy-600);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--navy-700);
    }
    
    /* ==========================================================================
       CUSTOM COMPONENT CLASSES
       ========================================================================== */
    
    /* Answer card */
    .answer-card {
        background-color: var(--navy-800);
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid var(--royal-blue);
        margin: 1rem 0;
    }
    
    .answer-card p {
        color: var(--text-primary);
        line-height: 1.7;
        margin: 0;
    }
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-badge.success {
        background-color: rgba(34, 197, 94, 0.15);
        color: var(--success);
    }
    
    .status-badge.warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: var(--warning);
    }
    
    .status-badge.error {
        background-color: rgba(239, 68, 68, 0.15);
        color: var(--error);
    }
    
    /* Info panel */
    .info-panel {
        background-color: var(--navy-800);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Section header */
    .section-header {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    st.title("📚 Answer IQ")
    st.caption("Retrieval-Augmented Generation Knowledge Base System")
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📝 Text Query", "🎤 Voice Query", "📋 History"])
    
    with tab1:
        render_text_query_tab()
    
    with tab2:
        render_voice_query_tab()
    
    with tab3:
        render_history_tab()


if __name__ == "__main__":
    main()
