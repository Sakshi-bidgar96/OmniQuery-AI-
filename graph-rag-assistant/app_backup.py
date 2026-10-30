import streamlit as st
import pypdf
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Graph RAG Assistant", layout="wide")
st.title("🧠 Graph RAG AI Knowledge Assistant")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    reader = pypdf.PdfReader(uploaded_file)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})

    st.success(f"PDF loaded successfully! {len(reader.pages)} pages")

    if not pages:
        st.warning("No text found. This may be a scanned PDF.")
        st.stop()

    # Split each page into smaller text chunks
    chunks = []
    for page in pages:
        words = page["text"].split()
        for start in range(0, len(words), 150):
            chunk = " ".join(words[start:start + 150])
            if chunk.strip():
                chunks.append({
                    "page": page["page"],
                    "text": chunk
                })

    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    st.subheader("📄 Extracted PDF Text")
    with st.expander("View extracted pages"):
        for page in pages:
            st.markdown(f"**Page {page['page']}**")
            st.write(page["text"][:2000])

    # Knowledge Graph
    st.subheader("🕸️ Knowledge Graph")

    graph_text = " ".join(texts)
    cv = CountVectorizer(
        stop_words="english",
        max_features=25,
        ngram_range=(1, 2)
    )
    counts = cv.fit_transform([graph_text])
    terms = cv.get_feature_names_out()
    freq = counts.toarray()[0]

    G = nx.Graph()
    for term, count in zip(terms, freq):
        if count >= 2:
            G.add_node(term, weight=int(count))

    # Connect terms that appear near each other
    for chunk in chunks:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b",
                           chunk["text"].lower())
        present = list(dict.fromkeys(
            w for w in words if w in G
        ))
        for i, a in enumerate(present):
            for b in present[i + 1:i + 4]:
                if a != b:
                    G.add_edge(a, b)

    if G.number_of_nodes() > 0:
        fig, ax = plt.subplots(figsize=(12, 7))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G, pos, ax=ax, with_labels=True,
            node_size=1400, font_size=8
        )
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Not enough text to build a graph.")

    # PDF Question Answering
    st.subheader("💬 Ask Your PDF")

    question = st.text_input("Type your question here")

    if st.button("Get Answer") and question.strip():
        q_vector = vectorizer.transform([question])
        scores = cosine_similarity(q_vector, matrix).flatten()
        best = scores.argsort()[::-1][:3]

        if scores[best[0]] <= 0:
            st.warning(
                "Relevant text was not found. Try another question."
            )
        else:
            st.markdown("### Relevant answer from your PDF")
            for idx in best:
                if scores[idx] > 0:
                    st.write(chunks[idx]["text"])
                    st.caption(
                        f"Source: PDF page {chunks[idx]['page']} "
                        f"| Relevance: {scores[idx]:.2f}"
                    )

            st.info(
                "This starter retrieves relevant PDF passages. "
                "It does not yet generate a conversational AI answer."
            )