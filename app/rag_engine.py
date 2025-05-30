import os
import json
from typing import List, Dict
from dotenv import load_dotenv
import boto3

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.bedrock_runtime = boto3.client("bedrock-runtime")
        self.model_id = "anthropic.claude-v2:1"
        self.temperature = 0.7
        self.max_tokens = 200
        self.top_p = 0.999
        self.top_k = 100
        self.stop_sequences = []

        self.conversation_history: List[Dict[str, str]] = []

    def build_message_content(self, context: str, question: str, conversation_history: str) -> str:
        """Build the text message content for Claude 3."""
        return f"""You are a helpful AI assistant. Use the following context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Previous conversation:
{conversation_history}

Context:
{context}

Question: {question}"""

    def generate_response(self, query: str, relevant_chunks: List[str]):
        context = "\n\n".join(relevant_chunks)

        conversation_text = "\n".join([
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in self.conversation_history[-5:]
        ])

        message_text = self.build_message_content(context, query, conversation_text)

        # Format the prompt in Claude's expected format
        prompt = f"\n\nHuman: {message_text}\n\nAssistant:"

        body = json.dumps({
            "prompt": prompt,
            "max_tokens_to_sample": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop_sequences": ["\n\nHuman:"],
            "anthropic_version": "bedrock-2023-05-31"
        })

        response = self.bedrock_runtime.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="*/*"
        )

        response_body = json.loads(response["body"].read())
        # Update to match Claude v2 response format
        answer = response_body.get("completion", "").strip()

        self.conversation_history.append({
            "question": query,
            "answer": answer
        })

        return answer


    def get_conversation_history(self):
        return self.conversation_history

    def clear_conversation_history(self):
        self.conversation_history = []