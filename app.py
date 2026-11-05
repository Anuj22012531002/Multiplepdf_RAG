import os
import time
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from evals import evaluate_rag_response
from guardrails import check_input_guardrails, check_output_guardrails

GEMINI_FLASH_CONTEXT_WINDOW = 1_048_576

st.set_page_config(
    page_title="Multi-PDF QnA with IR Metrics & Guardrails",
    page_icon="🛡️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def process_multiple_pdfs(uploaded_files, api_key):
    """Processes multiple uploaded PDFs and creates a unified FAISS index with document metadata."""
    all_split_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()

            # Attach original document filename to page metadata
            for doc in documents:
                doc.metadata["source_doc"] = uploaded_file.name

            split_docs = text_splitter.split_documents(documents)
            all_split_docs.extend(split_docs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # embeddings = OpenAIEmbeddings(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    vectorstore = FAISS.from_documents(all_split_docs, embeddings)
    return vectorstore


def generate_rag_response(query, vectorstore, api_key):
    start_time = time.perf_counter()

    # 1. Input Guardrails
    input_pass, input_violation = check_input_guardrails(query)
    if not input_pass:
        latency = round(time.perf_counter() - start_time, 3)
        return None, [], [], [], None, input_violation, {
            "latency": latency, "input_tokens": 0, "output_tokens": 0,
            "total_tokens": 0, "context_window": GEMINI_FLASH_CONTEXT_WINDOW,
            "remaining_context": GEMINI_FLASH_CONTEXT_WINDOW
        }

    # 2. Context Retrieval (k=4 across all documents)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.invoke(query)

    context_text = ""
    source_documents = set()
    source_pages_by_doc = {}
    doc_sources = []

    for idx, doc in enumerate(relevant_docs):
        doc_name = doc.metadata.get("source_doc", "Unknown Document")
        page_num = doc.metadata.get("page", 0) + 1
        
        context_text += f"[Doc: {doc_name}, Page: {page_num}]\n" + doc.page_content + "\n\n"
        source_documents.add(doc_name)
        
        if doc_name not in source_pages_by_doc:
            source_pages_by_doc[doc_name] = set()
        source_pages_by_doc[doc_name].add(page_num)

        doc_sources.append({
            "chunk_id": idx + 1,
            "doc_name": doc_name,
            "page": page_num,
            "content": doc.page_content
        })

    # Convert page sets to sorted lists
    formatted_sources = {doc: sorted(list(pages)) for doc, pages in source_pages_by_doc.items()}

    # 3. Instantiate LLM
    # llm = ChatOpenAI(model='gpt-4o-mini',api_key=api_key,temperature=0.2)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.2
    )

    full_prompt = f"""You are an intelligent assistant trained to answer questions strictly based on the provided context.
If the answer cannot be determined from the context, state that the information is not present in the documents.

Context:
{context_text}

Question:
{query}

Answer:"""

    # 4. Generate Response
    response = llm.invoke(full_prompt)
    raw_answer = response.content[0]['text']

    # 5. Token & Context Tracking
    input_tokens = llm.get_num_tokens(full_prompt)
    output_tokens = llm.get_num_tokens(raw_answer)
    total_tokens = input_tokens + output_tokens
    remaining_context = max(0, GEMINI_FLASH_CONTEXT_WINDOW - total_tokens)

    # 6. Output Guardrails
    output_pass, output_violation, clean_answer = check_output_guardrails(
        query=query,
        context=context_text,
        response=raw_answer,
        api_key=api_key
    )

    # 7. IR & Quality Metrics Computation
    eval_metrics = evaluate_rag_response(
        query=query,
        chunks=doc_sources,
        response=clean_answer,
        api_key=api_key
    )

    latency = round(time.perf_counter() - start_time, 3)
    
    perf_metrics = {
        "latency": latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "context_window": GEMINI_FLASH_CONTEXT_WINDOW,
        "remaining_context": remaining_context
    }

    violation_reason = output_violation if not output_pass else ""

    return clean_answer, list(source_documents), formatted_sources, doc_sources, eval_metrics, violation_reason, perf_metrics


def render_performance_and_token_metrics(perf):
    st.markdown("### ⚡ Execution Performance & Context Window")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(label="⏱️ Latency", value=f"{perf.get('latency', 0):.2f} s")
    with col2:
        st.metric(label="📥 Input Tokens", value=f"{perf.get('input_tokens', 0):,}")
    with col3:
        st.metric(label="📤 Output Tokens", value=f"{perf.get('output_tokens', 0):,}")
    with col4:
        st.metric(label="📊 Total Tokens", value=f"{perf.get('total_tokens', 0):,}")
    with col5:
        st.metric(label="📉 Remaining Context", value=f"{perf.get('remaining_context', 0):,}")

    st.caption(f"**Max Context Window Size:** {perf.get('context_window', 0):,} tokens")


def render_evaluation_dashboard(evals):
    st.markdown("### 📊 Advanced RAG & Information Retrieval (IR) Metrics")

    st.markdown("#### 🎯 Retrieval Metrics")
    r1_c1, r1_c2  = st.columns(2)
    
    k = evals.get("k", 4)
    with r1_c1:
        st.metric(label=f"Precision@{k}", value=f"{evals.get('precision_at_k', 0.0) * 100:.0f}%")
        st.progress(min(max(evals.get('precision_at_k', 0.0), 0.0), 1.0))

    with r1_c2:
        st.metric(label=f"Recall@{k}", value=f"{evals.get('recall_at_k', 0.0) * 100:.0f}%")
        st.progress(min(max(evals.get('recall_at_k', 0.0), 0.0), 1.0))


    st.divider()

    st.markdown("#### 🤖 Generation & Integrity Metrics")
    r2_c1, r2_c2, r2_c3, r2_c4, r2_c5 = st.columns(5)

    with r2_c1:
        h_rate = evals.get("hallucination_rate", 0.0)
        st.metric(label="Hallucination Rate", value=f"{h_rate * 100:.1f}%", delta=f"-{h_rate * 100:.1f}%" if h_rate > 0 else "0%", delta_color="inverse")
        st.progress(min(max(h_rate, 0.0), 1.0))
        st.caption(f"_{evals.get('hallucination_reasoning', '')}_")

    with r2_c2:
        f_score = evals.get("faithfulness", {}).get("score", 0.0)
        st.metric(label="Faithfulness", value=f"{f_score * 100:.0f}%")
        st.progress(min(max(f_score, 0.0), 1.0))

    with r2_c3:
        r_score = evals.get("recall", {}).get("score", 0.0)
        st.metric(label="Context Recall", value=f"{r_score * 100:.0f}%")
        st.progress(min(max(r_score, 0.0), 1.0))

    with r2_c4:
        rel_score = evals.get("relevance", {}).get("score", 0.0)
        st.metric(label="Answer Relevance", value=f"{rel_score * 100:.0f}%")
        st.progress(min(max(rel_score, 0.0), 1.0))

    with r2_c5:
        c_score = evals.get("correctness", {}).get("score", 0.0)
        st.metric(label="Correctness", value=f"{c_score * 100:.0f}%")
        st.progress(min(max(c_score, 0.0), 1.0))


# -----------------------------------------------------------------------------
# SIDEBAR UI
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")

    st.divider()
    st.header("📄 Multi-Document Upload")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if uploaded_files and api_key:
        current_filenames = sorted([f.name for f in uploaded_files])
        if st.session_state.loaded_files != current_filenames:
            with st.spinner("Processing PDFs and generating multi-doc embeddings..."):
                try:
                    st.session_state.vectorstore = process_multiple_pdfs(uploaded_files, api_key)
                    st.session_state.loaded_files = current_filenames
                    st.session_state.messages = []
                    st.success(f"Successfully processed **{len(uploaded_files)}** document(s)!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

    if st.session_state.loaded_files:
        st.caption("📚 **Loaded Documents:**")
        for fn in st.session_state.loaded_files:
            st.caption(f"- `{fn}`")

    st.divider()
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------------------------------------------------------
# MAIN CHAT INTERFACE
# -----------------------------------------------------------------------------
st.title("🛡️ Multi-PDF QnA Chatbot with IR Metrics & Guardrails")

if not api_key:
    st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
elif not uploaded_files:
    st.info("Please upload one or more PDF documents in the sidebar to start asking questions.")
else:
    # Render past chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant":
                if msg.get("guardrail_triggered"):
                    st.warning(f"⚠️ **Guardrail Warning:** {msg['guardrail_reason']}")

                if msg.get("source_docs"):
                    st.markdown("📄 **Source Document(s) Used:**")
                    for doc_name in msg["source_docs"]:
                        pages = msg.get("formatted_sources", {}).get(doc_name, [])
                        page_str = f" (Page {', '.join(map(str, pages))})" if pages else ""
                        st.caption(f"• **{doc_name}**{page_str}")
                
                if msg.get("docs"):
                    with st.expander("🔍 View Retrieved Source Chunks"):
                        for doc in msg["docs"]:
                            st.markdown(f"**Chunk {doc['chunk_id']} — Document: `{doc['doc_name']}` (Page {doc['page']})**")
                            st.text(doc["content"])
                            st.divider()

                if msg.get("perf_metrics"):
                    render_performance_and_token_metrics(msg["perf_metrics"])

                if msg.get("evals"):
                    render_evaluation_dashboard(msg["evals"])

    # User chat submission
    if user_query := st.chat_input("Ask a question across your documents..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing request, searching across documents & enforcing guardrails..."):
                try:
                    answer, source_docs, formatted_sources, doc_sources, eval_metrics, guardrail_reason, perf_metrics = generate_rag_response(
                        user_query,
                        st.session_state.vectorstore,
                        api_key
                    )

                    if answer is None:
                        blocked_msg = "I cannot answer this query as it violated safety guardrails."
                        st.markdown(blocked_msg)
                        st.error(f"⚠️ **Guardrail Warning:** {guardrail_reason}")
                        
                        if perf_metrics:
                            render_performance_and_token_metrics(perf_metrics)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": blocked_msg,
                            "guardrail_triggered": True,
                            "guardrail_reason": guardrail_reason,
                            "perf_metrics": perf_metrics
                        })
                    else:
                        st.markdown(answer)

                        if guardrail_reason:
                            st.warning(f"⚠️ **Guardrail Warning:** {guardrail_reason}")

                        if source_docs:
                            st.markdown("📄 **Source Document(s) Used:**")
                            for doc_name in source_docs:
                                pages = formatted_sources.get(doc_name, [])
                                page_str = f" (Page {', '.join(map(str, pages))})" if pages else ""
                                st.caption(f"• **{doc_name}**{page_str}")

                        if doc_sources:
                            with st.expander("🔍 View Retrieved Source Chunks"):
                                for doc in doc_sources:
                                    st.markdown(f"**Chunk {doc['chunk_id']} — Document: `{doc['doc_name']}` (Page {doc['page']})**")
                                    st.text(doc["content"])
                                    st.divider()

                        if perf_metrics:
                            render_performance_and_token_metrics(perf_metrics)

                        if eval_metrics:
                            render_evaluation_dashboard(eval_metrics)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "source_docs": source_docs,
                            "formatted_sources": formatted_sources,
                            "docs": doc_sources,
                            "evals": eval_metrics,
                            "perf_metrics": perf_metrics,
                            "guardrail_triggered": bool(guardrail_reason),
                            "guardrail_reason": guardrail_reason
                        })

                except Exception as e:
                    st.error(f"Failed to generate response: {e}")