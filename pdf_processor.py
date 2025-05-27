import boto3
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

class PDFProcessor:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.chunks = []

    def process_pdf(self, file_content: bytes, filename: str):
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name

        try:
            # Upload to S3
            self.s3_client.upload_file(
                temp_file_path,
                self.bucket_name,
                f"pdfs/{filename}"
            )

            # Process PDF with LangChain
            loader = PyPDFLoader(temp_file_path)
            pages = loader.load()

            # Split into chunks
            self.chunks = self.text_splitter.split_documents(pages)

        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    def get_chunks(self):
        return self.chunks