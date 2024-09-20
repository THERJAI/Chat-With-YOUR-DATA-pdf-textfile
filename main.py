from fastapi import FastAPI
from pydantic import BaseModel
import fitz  # PyMuPDF
import os
import uvicorn
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific origin(s) you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to the PDFs
pdf_paths = {
    1: "Patient_1.pdf",
    2: "Patient_2.pdf",
    3: "Patient_3.pdf",
    4: "Patient_4.pdf"
}

# Initialize global variables
query_engine = None


# Pre-initialize LLM and embeddings
@app.on_event("startup")
async def startup_event():
    global query_engine  # Declare it as global to use it in the endpoint
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI

    # Initialize the LLM model
    llm = HuggingFaceInferenceAPI(model_name='mistralai/Mistral-7B-Instruct-v0.3',
                                  token="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")

    # Set the Hugging Face API token as an environment variable
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    # Initialize the embeddings model
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    # Configure settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = None  # Disable node parsing for now
    Settings.num_output = 512
    Settings.context_window = 3900

# Define the Pydantic model for the request body
class PDFQuestionRequest(BaseModel):
    pdf_id: int
    question: str


# Define the API endpoint to handle the chat-like interaction
@app.post("/ask_question/")
async def ask_question(request: PDFQuestionRequest):
    global query_engine

    # Validate the PDF ID
    pdf_path = pdf_paths.get(request.pdf_id)
    if not pdf_path:
        return {"error": "Invalid PDF ID"}

    # Load the PDF document
    pdf_document = fitz.open(pdf_path)
    pdf_text = "".join([pdf_document.load_page(page_num).get_text() for page_num in range(pdf_document.page_count)])
    document = Document(text=pdf_text)

    # Create an index from the document
    index = VectorStoreIndex.from_documents([document], embed_model=Settings.embed_model)
    query_engine = index.as_query_engine()

    # Pass the user question to the query engine
    response = query_engine.query(request.question)

    # Return the response
    return {"response": response.response}


# Run the server
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")























































