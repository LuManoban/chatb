import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import requests
import os
import tempfile
import faiss
from langchain.docstore import InMemoryDocstore
import shutil
import pdfplumber  # Nueva librería para extraer tablas correctamente

# Configuración de OCR
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-24.08.0\Library\bin"

# Configuración de Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Directorios de almacenamiento
FAISS_INDEX_PATH = "./faiss_index"

# Cargar modelo de embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Cargar o crear FAISS
if os.path.exists(FAISS_INDEX_PATH):
    try:
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings=embedding_model,
                                        allow_dangerous_deserialization=True)
        st.sidebar.success("✅ FAISS cargado correctamente.")
    except Exception as e:
        st.warning("⚠️ Error al cargar FAISS. Se reiniciará la base de datos.")
        shutil.rmtree(FAISS_INDEX_PATH, ignore_errors=True)
        dimension = embedding_model.client.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dimension)
        vector_store = FAISS(index=index, embedding_function=embedding_model, docstore=InMemoryDocstore(),
                             index_to_docstore_id={})
else:
    dimension = embedding_model.client.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(index=index, embedding_function=embedding_model, docstore=InMemoryDocstore(),
                         index_to_docstore_id={})


# Extraer texto de PDFs mejorado
def extract_text_from_pdf(pdf_path):
    extracted_texts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                extracted_texts.append(f"Página {page_num + 1}: {text}")

            # Extraer tablas y estructurarlas correctamente
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    if table and len(table) > 1:
                        headers = [h.strip() if h else f"Columna_{i + 1}" for i, h in enumerate(table[0])]
                        structured_rows = []
                        for row in table[1:]:
                            row_data = {headers[i]: cell.strip() if cell else "" for i, cell in enumerate(row)}
                            structured_rows.append(row_data)

                        # Combinar filas relacionadas correctamente
                        merged_rows = []
                        temp_row = {}
                        for row in structured_rows:
                            if any(row.values()):  # Verifica si la fila tiene valores
                                if temp_row and row.get(headers[0]):  # Nueva fila válida
                                    merged_rows.append(temp_row)
                                    temp_row = row.copy()
                                else:  # Continuación de la fila anterior
                                    for key, value in row.items():
                                        if value:
                                            temp_row[key] = temp_row.get(key, "") + " " + value
                        if temp_row:
                            merged_rows.append(temp_row)

                        for row in merged_rows:
                            formatted_row = " | ".join(f"{key}: {value}" for key, value in row.items() if value)
                            extracted_texts.append(f"Tabla en página {page_num + 1}: {formatted_row}")

    # Si no se encontró texto, aplicar OCR
    if not extracted_texts:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        for page_num, img in enumerate(images):
            ocr_text = pytesseract.image_to_string(img, lang="spa").strip()
            if ocr_text:
                extracted_texts.append(f"Texto OCR Página {page_num + 1}: {ocr_text}")

    return extracted_texts


# Indexar PDFs
def index_pdfs(file_path, file_name):
    extracted_texts = extract_text_from_pdf(file_path)
    st.write("📜 Texto extraído del PDF:", extracted_texts)

    if extracted_texts:
        docs_with_metadata = [{"text": text, "metadata": {"source": file_name}} for text in extracted_texts]
        vector_store.add_texts(
            texts=[doc["text"] for doc in docs_with_metadata],
            metadatas=[doc["metadata"] for doc in docs_with_metadata]
        )
        vector_store.save_local(FAISS_INDEX_PATH)
        return f"✅ Documento '{file_name}' indexado en la base de datos."
    return "❌ No se pudo extraer texto del PDF."


# Consultar la base de datos y Ollama
def query_ollama(question):
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=5)
    st.write("🔍 Documentos recuperados:", docs_with_scores)

    if not docs_with_scores:
        return "No encontré información relevante en los documentos."

    threshold = 0.6  # Reducido para mejorar resultados
    relevant_texts = [doc.page_content for doc, score in docs_with_scores if score > threshold]

    if not relevant_texts:
        return "No encontré información suficientemente relevante."

    st.write("📜 Contexto enviado a Ollama:", relevant_texts)

    context = "\n".join(set(relevant_texts))
    prompt = f"Responde en español. Documento:\n{context}\n\nPregunta: {question}\nRespuesta:"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        return response.json().get("response", "No se obtuvo respuesta.")
    except requests.exceptions.RequestException as e:
        return f"Error al conectar con Ollama: {str(e)}"


# Interfaz Streamlit
st.title("📚 ChatBot con FAISS + OCR + Ollama")

# Subir e indexar documentos
uploaded_files = st.sidebar.file_uploader("📂 Sube tus PDFs", type="pdf", accept_multiple_files=True)

if st.sidebar.button("Cargar y procesar PDFs") and uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.getbuffer())
            pdf_path = temp_pdf.name
        status = index_pdfs(pdf_path, uploaded_file.name)
        os.remove(pdf_path)
        st.sidebar.write(status)
    st.sidebar.success("📂 Todos los archivos han sido indexados.")

# Preguntar sobre los documentos
question = st.text_input("Escribe tu pregunta sobre los documentos:")

if question:
    with st.spinner("Buscando información..."):
        answer = query_ollama(question)
    st.subheader("📢 Respuesta:")
    st.write(answer)

