"""Streamlit UI for the Hybrid SLM-LLM Financial Intelligence Pipeline.

Run with:
    streamlit run ui/app.py

Requires the FastAPI backend to be running at http://localhost:8000.
"""

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Financial Intelligence Pipeline",
    page_icon="🏦",
    layout="wide",
)

# ── Sidebar navigation ─────────────────────────────────────────────
page = st.sidebar.radio(
    "Navigation",
    ["🔍 Analyze", "📚 Documents", "📊 Metrics"],
)

# Check backend health
try:
    health = requests.get(f"{API_BASE}/health", timeout=3)
    if health.status_code == 200:
        st.sidebar.success("Backend: Connected")
    else:
        st.sidebar.error("Backend: Unhealthy")
except requests.ConnectionError:
    st.sidebar.error("Backend: Offline")
    st.sidebar.caption("Start with: `uvicorn app.main:app --reload`")


# ════════════════════════════════════════════════════════════════════
# PAGE: Analyze
# ════════════════════════════════════════════════════════════════════
if page == "🔍 Analyze":
    st.title("🏦 Financial Text Analysis")
    st.caption("Route financial text to the optimal model — SLM for simple tasks, LLM for complex ones.")

    # --- Input form ---
    col_input, col_options = st.columns([3, 1])

    with col_input:
        financial_text = st.text_area(
            "Financial Text",
            height=200,
            placeholder=(
                "Paste financial text here...\n\n"
                "Example: Apple reported Q3 2024 revenue of $81.8 billion, "
                "up 5% year-over-year. Services revenue hit an all-time high "
                "of $24.2 billion."
            ),
        )

    with col_options:
        task_type = st.selectbox(
            "Task Type",
            [
                "summarization",
                "extraction",
                "classification",
                "sentiment",
                "risk_analysis",
                "trend_analysis",
                "reasoning",
                "multi_step",
                "comparison",
            ],
        )

        st.divider()

        use_rag = st.checkbox("Use RAG Context", value=False)
        rag_top_k = 3
        if use_rag:
            rag_top_k = st.slider("Context chunks (top-k)", 1, 10, 3)

    # --- Submit ---
    if st.button("Analyze", type="primary", use_container_width=True):
        if not financial_text.strip():
            st.warning("Please enter some financial text.")
        else:
            with st.spinner("Running inference..."):
                try:
                    resp = requests.post(
                        f"{API_BASE}/analyze",
                        json={
                            "financial_text": financial_text,
                            "task_type": task_type,
                            "use_rag": use_rag,
                            "rag_top_k": rag_top_k,
                        },
                        timeout=120,
                    )
                    if resp.status_code != 200:
                        st.error(f"API error {resp.status_code}: {resp.text}")
                    else:
                        data = resp.json()

                        # --- Result metrics row ---
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Model", data["selected_model"])
                        m2.metric("Routing", data["routing_decision"])
                        m3.metric("Confidence", f"{data['confidence_score']:.2%}")
                        m4.metric("Latency", f"{data['latency_ms']:.0f} ms")
                        m5.metric("Cost", f"${data['estimated_cost_usd']:.6f}")

                        # --- Extra info row ---
                        e1, e2, e3 = st.columns(3)
                        e1.metric("Tokens Used", data.get("tokens_used", 0))
                        e2.metric("Cache Hit", "✅ Yes" if data.get("cache_hit") else "❌ No")
                        e3.metric("RAG Sources", data.get("rag_sources_used", 0))

                        # --- Response ---
                        st.divider()
                        st.subheader("Analysis Result")
                        st.write(data["final_response"])

                        # --- Request ID ---
                        st.caption(f"Request ID: `{data.get('request_id', 'N/A')}`")

                except requests.ConnectionError:
                    st.error("Cannot connect to backend. Is it running?")
                except Exception as e:
                    st.error(f"Error: {e}")


# ════════════════════════════════════════════════════════════════════
# PAGE: Documents
# ════════════════════════════════════════════════════════════════════
elif page == "📚 Documents":
    st.title("📚 Document Store")
    st.caption("Ingest financial documents for RAG-powered analysis.")

    tab_ingest, tab_upload, tab_search = st.tabs(["Ingest Text", "Upload File", "Search"])

    # --- Tab: Ingest text ---
    with tab_ingest:
        doc_title = st.text_input("Title", placeholder="Apple 10-K FY2024")
        doc_source = st.text_input("Source", placeholder="SEC EDGAR")
        doc_text = st.text_area(
            "Document Text",
            height=200,
            placeholder="Paste the full document text here...",
        )

        if st.button("Ingest Document", type="primary"):
            if not doc_text.strip():
                st.warning("Please enter document text.")
            else:
                with st.spinner("Chunking and embedding..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/documents/ingest",
                            json={
                                "text": doc_text,
                                "title": doc_title,
                                "source": doc_source,
                            },
                            timeout=60,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.success(
                                f"Ingested! Doc ID: `{data['doc_id']}` — "
                                f"{data['chunks_stored']} chunks stored."
                            )
                        else:
                            st.error(f"Error: {resp.text}")
                    except requests.ConnectionError:
                        st.error("Cannot connect to backend.")

    # --- Tab: Upload file ---
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a .txt or .pdf file",
            type=["txt", "pdf"],
        )
        upload_title = st.text_input("Title (optional)", key="upload_title")
        upload_source = st.text_input("Source (optional)", key="upload_source")

        if st.button("Upload & Ingest") and uploaded_file:
            with st.spinner("Processing file..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    params = {}
                    if upload_title:
                        params["title"] = upload_title
                    if upload_source:
                        params["source"] = upload_source

                    resp = requests.post(
                        f"{API_BASE}/documents/upload",
                        files=files,
                        params=params,
                        timeout=60,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(
                            f"Uploaded! Doc ID: `{data['doc_id']}` — "
                            f"{data['chunks_stored']} chunks stored."
                        )
                    else:
                        st.error(f"Error: {resp.text}")
                except requests.ConnectionError:
                    st.error("Cannot connect to backend.")

    # --- Tab: Search ---
    with tab_search:
        query = st.text_input("Search query", placeholder="revenue growth margins")
        search_top_k = st.slider("Results to return", 1, 10, 3, key="search_k")

        if st.button("Search") and query:
            with st.spinner("Searching..."):
                try:
                    resp = requests.get(
                        f"{API_BASE}/documents/search",
                        params={"query": query, "top_k": search_top_k},
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data["total_results"] == 0:
                            st.info("No results found. Ingest some documents first.")
                        else:
                            for i, result in enumerate(data["results"], 1):
                                with st.expander(
                                    f"Result {i} — Score: {result['score']:.4f} "
                                    f"(Doc: {result['doc_id']})"
                                ):
                                    st.write(result["text"])
                                    st.caption(
                                        f"Title: {result['metadata'].get('title', 'N/A')} | "
                                        f"Source: {result['metadata'].get('source', 'N/A')}"
                                    )
                    else:
                        st.error(f"Error: {resp.text}")
                except requests.ConnectionError:
                    st.error("Cannot connect to backend.")

    # --- Document list ---
    st.divider()
    st.subheader("Ingested Documents")

    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=10)
        if resp.status_code == 200:
            docs = resp.json()
            if not docs:
                st.info("No documents ingested yet.")
            else:
                for doc in docs:
                    col_info, col_action = st.columns([4, 1])
                    with col_info:
                        st.markdown(
                            f"**{doc['title'] or 'Untitled'}** — "
                            f"`{doc['doc_id']}` — "
                            f"{doc['total_chunks']} chunks — "
                            f"Source: {doc['source'] or 'N/A'}"
                        )
                    with col_action:
                        if st.button("🗑️ Delete", key=f"del_{doc['doc_id']}"):
                            try:
                                del_resp = requests.delete(
                                    f"{API_BASE}/documents/{doc['doc_id']}",
                                    timeout=10,
                                )
                                if del_resp.status_code == 200:
                                    st.success("Deleted!")
                                    st.rerun()
                                else:
                                    st.error(f"Error: {del_resp.text}")
                            except requests.ConnectionError:
                                st.error("Cannot connect to backend.")
        else:
            st.error("Failed to load documents.")
    except requests.ConnectionError:
        st.warning("Cannot connect to backend.")


# ════════════════════════════════════════════════════════════════════
# PAGE: Metrics
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Metrics":
    st.title("📊 Pipeline Metrics")
    st.caption("Real-time observability data from the inference pipeline.")

    if st.button("🔄 Refresh"):
        st.rerun()

    try:
        resp = requests.get(f"{API_BASE}/metrics", timeout=10)
        if resp.status_code == 200:
            m = resp.json()

            # --- Request stats ---
            st.subheader("Requests")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Requests", m.get("total_requests", 0))
            c2.metric("SLM Requests", m.get("slm_requests", 0))
            c3.metric("LLM Requests", m.get("llm_requests", 0))
            c4.metric("Fallbacks", m.get("fallback_count", 0))

            # --- Rates ---
            st.subheader("Rates & Latency")
            r1, r2, r3 = st.columns(3)
            r1.metric("Fallback Rate", f"{m.get('fallback_rate', 0):.2%}")
            r2.metric("Avg Latency", f"{m.get('avg_latency_ms', 0):.1f} ms")
            r3.metric("Cache Hit Rate", f"{m.get('cache_hit_rate', 0):.2%}")

            # --- Cache ---
            st.subheader("Cache")
            ca1, ca2 = st.columns(2)
            ca1.metric("Cache Hits", m.get("cache_hits", 0))
            ca2.metric("Cache Misses", m.get("cache_misses", 0))

            # --- Cost ---
            st.subheader("Cost")
            co1, co2, co3 = st.columns(3)
            co1.metric("Total Tokens", f"{m.get('total_tokens_used', 0):,}")
            co2.metric(
                "Total Cost",
                f"${m.get('total_estimated_cost_usd', 0):.6f}",
            )
            co3.metric(
                "Saved vs GPT-4o",
                f"${m.get('total_cost_saved_vs_gpt4o', 0):.6f}",
            )

            # --- Raw JSON ---
            with st.expander("Raw JSON"):
                st.json(m)
        else:
            st.error("Failed to fetch metrics.")
    except requests.ConnectionError:
        st.warning("Cannot connect to backend. Start it first.")
