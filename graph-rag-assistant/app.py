
import os
import hashlib

import streamlit as st
import pypdf
import chromadb
from chromadb.utils import embedding_functions


# APP CONFIGURATION
st.set_page_config(
    page_title="OmniQuery AI",
    page_icon="✨",
    layout="wide",
)

st.title("✨ OmniQuery AI")
st.subheader("An All-in-One Question Answering Assistant")

st.write(
    "Upload multiple PDF documents, search their content, "
    "and get clear answers using AI."
)

# DATABASE SETTINGS
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pdf_knowledge"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

# =========================================================
# 2. INITIALIZE CHROMADB
# =========================================================

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    embedding_function = (
        embedding_functions.DefaultEmbeddingFunction()
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    return collection


# =========================================================
# 3. EXTRACT TEXT FROM PDF
# =========================================================

def extract_pdf_pages(pdf_file):
    pdf_file.seek(0)

    reader = pypdf.PdfReader(pdf_file)

    pages = []

    for page_number, page in enumerate(
        reader.pages, start=1
    ):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip(),
            })

    return pages


# =========================================================
# 4. SPLIT TEXT INTO CHUNKS
# =========================================================

def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP,
):
    text = " ".join(text.split())

    if not text:
        return []

    if overlap >= chunk_size:
        overlap = chunk_size // 5

    chunks = []
    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text),
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# =========================================================
# 5. CREATE STABLE CHUNK ID
# =========================================================

def create_chunk_id(
    filename,
    page_number,
    chunk_number,
    chunk_text,
):
    raw_text = (
        f"{filename}|{page_number}|"
        f"{chunk_number}|{chunk_text}"
    )

    return hashlib.sha256(
        raw_text.encode("utf-8")
    ).hexdigest()


# =========================================================
# 6. INDEX ONE PDF
# =========================================================

def index_pdf(pdf_file, collection):
    filename = os.path.basename(pdf_file.name)

    pages = extract_pdf_pages(pdf_file)

    documents = []
    metadatas = []
    ids = []

    for page in pages:
        chunks = split_text(page["text"])

        for chunk_number, chunk in enumerate(chunks):
            document_id = create_chunk_id(
                filename,
                page["page"],
                chunk_number,
                chunk,
            )

            documents.append(chunk)

            metadatas.append({
                "source": filename,
                "page": int(page["page"]),
                "chunk": int(chunk_number),
            })

            ids.append(document_id)

    if documents:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

    return {
        "filename": filename,
        "pages": len(pages),
        "chunks": len(documents),
    }


# =========================================================
# 7. SEARCH RELEVANT DOCUMENT PASSAGES
# =========================================================

def search_documents(
    question,
    collection,
    top_k=DEFAULT_TOP_K,
):
    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    passages = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        metadata = metadata or {}

        passages.append({
            "text": document,
            "source": metadata.get(
                "source", "Unknown PDF"
            ),
            "page": metadata.get("page", "?"),
            "distance": distance,
        })

    return passages


# =========================================================
# 8. GENERATE ANSWER USING OLLAMA
# =========================================================

def generate_answer(
    question,
    passages,
    model_name,
):
    if not passages:
        return (
            "Relevant text was not found in the "
            "indexed PDFs. Try another question."
        )

    context_parts = []

    for index, passage in enumerate(
        passages, start=1
    ):
        context_parts.append(
            f"[Source {index}: "
            f"{passage['source']}, "
            f"page {passage['page']}]\n"
            f"{passage['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a document-based AI learning assistant.

Answer the user's question using only the supplied
document passages.

Rules:
1. Do not invent facts.
2. If evidence is insufficient, say so clearly.
3. Use simple, clear language.
4. Cite the PDF filename and page number.
5. Do not claim that a source supports something
   unless the supplied passage contains that evidence.

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        import requests

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        response.raise_for_status()

        answer = response.json().get(
            "response", ""
        ).strip()

        if answer:
            return answer

    except Exception:
        pass

    # Fallback: show the retrieved evidence instead
    # of pretending an LLM answer was generated.
    evidence = []

    for passage in passages:
        evidence.append(
            f"**{passage['source']} — "
            f"Page {passage['page']}**\n\n"
            f"{passage['text']}"
        )

    return (
        "Ollama could not be reached. "
        "The relevant passages are shown below:\n\n"
        + "\n\n---\n\n".join(evidence)
    )


# =========================================================
# 9. SIDEBAR SETTINGS
# =========================================================

with st.sidebar:
    st.header("⚙️ Settings")

    ollama_model = st.text_input(
        "Ollama model name",
        value="llama3.2",
    )

    top_k = st.slider(
        "Number of passages to retrieve",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    st.caption(
        "Generated answers require Ollama to be "
        "running locally with the selected model."
    )


# =========================================================
# 10. CONNECT TO CHROMADB
# =========================================================

try:
    collection = get_collection()

except Exception as error:
    st.error(
        f"Could not initialize ChromaDB: {error}"
    )
    st.stop()


# =========================================================
# 11. UPLOAD MULTIPLE PDF FILES
# =========================================================

st.header("1. Upload PDF Files")

uploaded_files = st.file_uploader(
    "Select one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(
        f"Selected PDF files: {len(uploaded_files)}"
    )

    if st.button(
        "📚 Index Selected PDFs",
        type="primary",
    ):
        for pdf_file in uploaded_files:
            try:
                with st.spinner(
                    f"Processing {pdf_file.name}..."
                ):
                    result = index_pdf(
                        pdf_file,
                        collection,
                    )

                if result["chunks"] > 0:
                    st.success(
                        f"Indexed {result['filename']} | "
                        f"Text pages: {result['pages']} | "
                        f"Chunks: {result['chunks']}"
                    )
                else:
                    st.warning(
                        f"No selectable text found in "
                        f"{result['filename']}. "
                        "Scanned PDFs may require OCR."
                    )

            except Exception as error:
                st.error(
                    f"Error processing "
                    f"{pdf_file.name}: {error}"
                )

        st.success("PDF processing finished.")


# =========================================================
# 12. SHOW INDEXED DOCUMENTS
# =========================================================

st.header("2. Indexed Document Inventory")

try:
    stored_data = collection.get(
        include=["metadatas"]
    )

    inventory = {}

    for metadata in (
        stored_data.get("metadatas") or []
    ):
        if metadata:
            filename = metadata.get(
                "source", "Unknown PDF"
            )

            inventory[filename] = (
                inventory.get(filename, 0) + 1
            )

    st.metric(
        "Total Indexed Chunks",
        collection.count(),
    )

    if inventory:
        st.dataframe(
            [
                {
                    "PDF File": filename,
                    "Indexed Chunks": chunk_count,
                }
                for filename, chunk_count
                in sorted(inventory.items())
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No PDFs indexed yet. Upload PDFs above."
        )

except Exception as error:
    st.warning(
        f"Could not load document inventory: {error}"
    )


# =========================================================
# 13. ASK QUESTIONS
# =========================================================

st.header("3. Ask a Question")

question = st.text_input(
    "Enter your question about the PDFs",
    placeholder="What is the main topic of the document?",
)

if st.button("🔍 Search and Answer"):
    if not question.strip():
        st.warning(
            "Please enter a question."
        )

    elif collection.count() == 0:
        st.warning(
            "Upload and index at least one PDF first."
        )

    else:
        with st.spinner(
            "Searching relevant passages..."
        ):
            passages = search_documents(
                question.strip(),
                collection,
                top_k,
            )

        st.subheader("Answer")

        answer = generate_answer(
            question.strip(),
            passages,
            ollama_model,
        )

        st.markdown(answer)

        st.subheader("Retrieved Evidence")

        if passages:
            for index, passage in enumerate(
                passages, start=1
            ):
                with st.expander(
                    f"Source {index}: "
                    f"{passage['source']} | "
                    f"Page {passage['page']} | "
                    f"Cosine distance: "
                    f"{passage['distance']:.4f}"
                ):
                    st.write(passage["text"])

        else:
            st.warning(
                "Relevant text was not found. "
                "Try another question."
            )


# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

st.caption(
    "OmniQuery AI | "
    "PDF extraction + ChromaDB semantic retrieval "
    "+ Ollama AI answer generation."
)


st.caption(
    "This standalone version does not include "
    "Neo4j graph extraction or graph visualization."
)