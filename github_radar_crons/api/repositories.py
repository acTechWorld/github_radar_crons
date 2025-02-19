import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
base_url = os.getenv('GITHUB_BACKEND_BASED_URL')

def fetch_repositories(params):

    # Combine the base URL and API path
    full_url = f"{base_url}/api/repositories"
    # Send a GET request to the external API with the parameters
    response = requests.get(full_url, params=params)

    # If the request is successful (status code 200), return the JSON data
    if response.status_code == 200:
        return response.json()  # Return the data as JSON
    else:
        # If the request fails, return an error message
        print(f"Error: Failed to fetch data from the API fetch_repositories. Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")

def saveAIContent(jsonBody):
    # Combine the base URL and API path
    full_url = f"{base_url}/api/aiContents"
    # Send a GET request to the external API with the parameters
    response = requests.post(full_url, json=jsonBody)

    if response.status_code == 201:
        return response.json()  # Return the data as JSON
    else:
        # If the request fails, return an error message
        print(f"Error: Failed to fetch data from the API saveAIContent. Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")