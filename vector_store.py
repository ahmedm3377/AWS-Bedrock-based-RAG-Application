import os
import json
import boto3
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
from dotenv import load_dotenv

load_dotenv()

class VectorStore:
    def __init__(self):
        # Initialize Pinecone
        self.pinecone_ = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "rag-index"
        self.dimension = 1536  # Titan embedding dimension

        # Ensure index exists
        if not self.pinecone_.has_index(self.index_name):
            self.pinecone_.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=Metric.COSINE,
                spec=ServerlessSpec(
                    cloud=CloudProvider.AWS,
                    region=AwsRegion.US_EAST_1,
                )
            )

        self.index = self.pinecone_.Index(self.index_name)

        # Bedrock client
        self.bedrock = boto3.client("bedrock-runtime")
        self.model_id = "amazon.titan-embed-text-v1"

    def get_embedding(self, text: str) -> list:
        body = {
            "inputText": text
        }

        response = self.bedrock.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        return result["embedding"]

    def store_vectors(self, chunks):
        # Assume chunks is a list of dicts: [{'text': ...}, ...]
        vectors = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            embedding = self.get_embedding(text)
            vectors.append({
                "id": f"vec_{i}",
                "values": embedding,
                "metadata": {"text": text}
            })
        self.index.upsert(vectors=vectors)

    def search(self, query: str, top_k: int = 5):
        query_embedding = self.get_embedding(query)

        try:
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer")

            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            matches = results.get("matches", [])
            return [
                match.get("metadata", {}).get("text", "")
                for match in matches
            ]
        except Exception as e:
            raise ValueError(f"Error during vector search: {str(e)}")
