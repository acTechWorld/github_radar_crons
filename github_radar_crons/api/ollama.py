import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def chat_mono(message):

    # Send a GET request to the API
    base_url = os.getenv('OLLAMA_BASED_URL')
    # Combine the base URL and API path
    full_url = f"{base_url}/api/chat"

    bodyMessage = {
        "model": "deepseek-r1:14b",
        "messages": [{
            "role": "user",
            "content": message
        }],
        "stream": False
    }
    # Send a GET request to the external API with the parameters
    response = requests.post(full_url, json=bodyMessage)

    # If the request is successful (status code 200), return the JSON data
    if response.status_code == 200:
        return response.json()  # Return the data as JSON
    else:
        # If the request fails, return an error message
        print(f"Error: Failed to fetch data from the API. Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")


def chat_multi(messages):
    # Send a GET request to the API
    base_url = os.getenv('OLLAMA_BASED_URL')
    # Combine the base URL and API path
    full_url = f"{base_url}/api/chat"

    bodyMessage = {
        #"model": "llama3.1:8b",
        "model": "llama3.1:8b_32K", 
        "messages": messages,
        "stream": False
    }
    # Send a GET request to the external API with the parameters
    response = requests.post(full_url, json=bodyMessage)

    # If the request is successful (status code 200), return the JSON data
    if response.status_code == 200:
        return response.json()  # Return the data as JSON
    else:
        # If the request fails, return an error message
        print(f"Error: Failed to fetch data from the API. Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")