import os
import requests
import json
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq API endpoint
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def summarize_conversation(conversation):
    """Summarizes the customer support conversation using Groq API."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "Summarize the following customer support conversation in a few sentences."},
            {"role": "user", "content": conversation}
        ],
        "temperature": 0.7
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"Error in summarization: {response.status_code}, {response.text}"

def generate_improvements(conversation):
    """Suggests better agent responses using Groq API."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""Analyze the following conversation and suggest improvements for the agent's responses:
    
    {conversation}
    
    Provide a list of original agent responses and improved responses."""

    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "Suggest improvements for the agent's responses in a customer support conversation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        improved_responses = response.json()["choices"][0]["message"]["content"]
        return improved_responses.split("\n\n")  # Splitting into list items
    else:
        return f"Error in generating improvements: {response.status_code}, {response.text}"
