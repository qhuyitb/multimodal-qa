"""
Simple Streamlit QA App
Upload documents and ask questions - that's it!
"""

import streamlit as st
import re
from typing import List

# Page config
st.set_page_config(
    page_title="Document QA",
    page_icon="�",
    layout="centered"
)

# Initialize components lazily
@st.cache_resource
def get_retriever():
    """Create hybrid retriever (cached)"""
    from src.services.hybrid_retrieval import HybridRetriever
    retriever = HybridRetriever(
        embedding_model="paraphrase-multilingual-mpnet-base-v2",
        collection_name="streamlit_qa"
    )
    return retriever

@st.cache_resource
def get_qa_service():
    """Create QA service (cached)"""
    try:
        from src.services.generative_qa import create_generative_qa_service
        from src.services.adaptive_qa import create_adaptive_qa_service
        
        extractive_qa = create_adaptive_qa_service(
            model_path="models/xlm_roberta_qa/stage2_best",
            device="cpu",
            default_language="vi"
        )
        
        generative_qa = create_generative_qa_service(
            extractive_qa_service=extractive_qa,
            model_name="qwen2.5:7b",
            ollama_base_url="http://localhost:11434"
        )
        return generative_qa, "generative"
    except Exception as e:
        st.warning(f"Ollama unavailable, using extractive QA only: {e}")
        from src.services.adaptive_qa import create_adaptive_qa_service
        extractive_qa = create_adaptive_qa_service(
            model_path="models/xlm_roberta_qa/stage2_best",
            device="cpu",
            default_language="vi"
        )
        return extractive_qa, "extractive"


def simple_chunks(text: str, chunk_size: int = 500) -> List[str]:
    """Split text into chunks"""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [normalized[i:i + chunk_size] for i in range(0, len(normalized), chunk_size)]


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file"""
    filename = uploaded_file.name.lower()
    content = uploaded_file.read()
    
    if filename.endswith('.txt'):
        return content.decode('utf-8')
    
    elif filename.endswith('.pdf'):
        import pdfplumber
        import io
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
    
    elif filename.endswith('.docx'):
        import docx
        import io
        doc = docx.Document(io.BytesIO(content))
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    return ""


def index_file(retriever, filename: str, text: str) -> int:
    """Index a file into the retriever"""
    chunks = simple_chunks(text)
    if not chunks:
        return 0
    
    import uuid
    from datetime import datetime
    
    base_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    chunk_ids = [f"{base_id}_chunk_{i}" for i in range(len(chunks))]
    chunk_metadata = [{
        "filename": filename,
        "chunk_index": i,
        "total_chunks": len(chunks)
    } for i in range(len(chunks))]
    
    retriever.index_documents(
        documents=chunks,
        ids=chunk_ids,
        metadata=chunk_metadata,
        append=True
    )
    
    return len(chunks)


# Session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("📄 Document QA")
st.caption("Upload documents, ask questions - simple as that!")

# Sidebar - Upload
with st.sidebar:
    st.header("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Upload TXT, PDF, or DOCX files"
    )
    
    if uploaded_files:
        new_files = []
        for uf in uploaded_files:
            file_key = f"{uf.name}_{uf.size}"
            if file_key not in [f['key'] for f in st.session_state.uploaded_files]:
                new_files.append((uf, file_key))
        
        if new_files:
            with st.spinner("Processing files..."):
                retriever = get_retriever()
                
                for uf, file_key in new_files:
                    try:
                        text = extract_text_from_file(uf)
                        if text.strip():
                            chunks = index_file(retriever, uf.name, text)
                            st.session_state.uploaded_files.append({
                                'key': file_key,
                                'name': uf.name,
                                'chunks': chunks,
                                'chars': len(text)
                            })
                            st.success(f"{uf.name}: {chunks} chunks")
                        else:
                            st.warning(f"{uf.name}: empty or unreadable")
                    except Exception as e:
                        st.error(f"{uf.name}: {str(e)}")
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown("### Indexed Files")
        total_chunks = 0
        for f in st.session_state.uploaded_files:
            st.caption(f"• {f['name']} ({f['chunks']} chunks)")
            total_chunks += f['chunks']
        st.info(f"Total: {len(st.session_state.uploaded_files)} files, {total_chunks} chunks")
        
        if st.button("Clear All", use_container_width=True):
            retriever = get_retriever()
            retriever.clear()
            st.session_state.uploaded_files = []
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Upload documents to start")

# Main area - Chat
st.markdown("---")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.uploaded_files:
        st.warning("Please upload documents first!")
    else:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    retriever = get_retriever()
                    qa_service, qa_type = get_qa_service()
                    
                    # Retrieve context
                    results = retriever.hybrid_search(prompt, top_k=5, alpha=0.7)
                    
                    if not results:
                        answer = "Không tìm thấy thông tin liên quan trong tài liệu đã upload."
                    else:
                        # Build context
                        context = "\n\n".join([r['document'] for r in results])
                        
                        # Get answer
                        response = qa_service.answer(
                            question=prompt,
                            context=context
                        )
                        answer = response.answer if hasattr(response, 'answer') else response.get('answer', 'Không có câu trả lời')
                        
                        # Add sources
                        sources = []
                        for r in results[:3]:
                            meta = r.get('metadata', {})
                            sources.append(f"- {meta.get('filename', 'Unknown')}")
                        
                        if sources:
                            answer += f"\n\n**Nguồn:** {', '.join(set(sources))}"
                    
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.caption("Tip: Upload multiple files and ask questions in Vietnamese or English!")
