from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
from dotenv import load_dotenv
from app.pdf_processor import PDFProcessor
from app.vector_store import VectorStore
from app.rag_engine import RAGEngine

# Load environment variables
load_dotenv()
import  logging


logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Application")

# # Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pdf_processor_cls = PDFProcessor()
vector_store_cls = VectorStore()
rag_engine_cls = RAGEngine()

class QueryRequest(BaseModel):
    query: str

class ConversationItem(BaseModel):
    question: str
    answer: str

@app.get("/health")
async def health_check():
    logger.info("Health check successful")
    return {"status": "healthy"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Process and store PDF
        file_content = await file.read()
        pdf_processor_cls.process_pdf(file_content, file.filename)

        # Generate embeddings and store in Pinecone
        chunks = pdf_processor_cls.get_chunks()
        vector_store_cls.store_vectors(chunks)

        # Clear conversation history when new document is uploaded
        rag_engine_cls.clear_conversation_history()

        return {"message": "PDF processed and stored successfully"}
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    try:
        # Get relevant chunks from vector store
        relevant_chunks = vector_store_cls.search(request.query)

        # Generate response using RAG
        response = rag_engine_cls.generate_response(request.query, relevant_chunks)

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation")
async def get_conversation():
    try:
        history = rag_engine_cls.get_conversation_history()
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/conversation")
async def clear_conversation():
    try:
        rag_engine_cls.clear_conversation_history()
        return {"message": "Conversation history cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# # AWS Lambda handler
lambda_handler = Mangum(app)