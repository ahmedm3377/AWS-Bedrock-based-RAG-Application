# AWS Bedrock-based RAG Application

A FastAPI-based Retrieval-Augmented Generation (RAG) application that processes PDF documents and provides intelligent responses using AWS Bedrock and Pinecone vector database.

## Features

- PDF document processing and chunking usning LangChain
- Vector embeddings generation using Amazon Titan
- Semantic search capabilities with Pinecone
- Conversational AI responses using Claude v2
- RESTful API endpoints for document upload and querying
- Conversation history management

## Tech Stack

- **FastAPI** - Modern web framework for building APIs
- **AWS Bedrock** - For LLM integration (Claude v2) and embeddings (Titan)
- **Pinecone** - Serverless vector database for semantic search
- **LangChain** - For PDF processing and document chunking
- **Boto3** - AWS SDK for Python
- **Mangum** - AWS Lambda handler for FastAPI

## AWS Bedrock Integration

The application leverages two key AWS Bedrock models:

### 1. Amazon Titan Embeddings Model
- **Model ID**: `amazon.titan-embed-text-v1`
- **Purpose**: Generates text embeddings for document chunks and queries
- **Features**:
  - 1536-dimensional embeddings
  - Optimized for semantic similarity search
  - Used for both document indexing and query processing

### 2. Anthropic Claude v2
- **Model ID**: `anthropic.claude-v2:1`
- **Purpose**: Generates conversational responses using RAG
- **Configuration**:
  - Temperature: 0.7 (balanced between creativity and accuracy)
  - Max tokens: 200
  - Top-p: 0.999
  - Top-k: 100
- **Features**:
  - Conversation history tracking
  - Context-aware responses
  - Maintains conversation coherence

### RAG Pipeline Flow

1. **Document Processing**:
   ```text
   Upload PDF → Split into chunks → Generate Titan embeddings → Store in Pinecone
   ```

2. **Query Processing**:
   ```text
   User query → Generate query embedding → Search Pinecone → Retrieve relevant chunks → Claude generates response
   ```

## API Endpoints

### Health Check
http GET /health

Returns the health status of the application.

### Upload PDF
http POST /upload

Upload and process a PDF document. The document will be:
- Stored in S3
- Chunked into segments
- Converted to embeddings using Titan
- Stored in Pinecone vector database

### Query
```http
POST /query
Content-Type: application/json

{
    "query": "your question here"
}
```
Processes queries through the RAG pipeline:
1. Converts query to embedding using Titan
2. Retrieves relevant context from Pinecone
3. Generates response using Claude v2

### Conversation Management
```http
GET /conversation
```
Retrieve conversation history.

```http
DELETE /conversation
```
Clear conversation history.

## Environment Variables

The following environment variables are required:

- `S3_BUCKET_NAME` - AWS S3 bucket name
- `PINECONE_API_KEY` - Pinecone API key

## AWS Bedrock Setup

1. **Enable AWS Bedrock Access**:
   - Navigate to AWS Console
   - Go to AWS Bedrock service --> Model Access
   - Enable access to Anthropic Claude and Amazon Titan models

2. **IAM Configuration AWS Lambda **:
   - Create an IAM role with permissions to access S3 bucket and Bedrock:

## Setup and Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install dependencies:
```aiignore
pip install -r requirements.txt
```
- For local run
```aiignore
uvicorn main:app --reload
```