import os
import requests

class LlmHandler:
    def __init__(self):
        self.HF_TOKEN = os.getenv("HF_TOKEN", "...your_huggingface_token_here...")
        self.API_URL = "https://router.huggingface.co/v1/chat/completions"
        self.MODEL = "meta-llama/Llama-3.1-8B-Instruct:fireworks-ai"

    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """
        Send a prompt to the Hugging Face Chat API and receive a response
        """
        headers = {
            "Authorization": f"Bearer {self.HF_TOKEN}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": max_length
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating response: {e}"

    def ask(self, prompt: str):
        """
        Alias cho generate_response
        """
        return self.generate_response(prompt)
