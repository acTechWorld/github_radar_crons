import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def fetch_repositories(params):

    # Send a GET request to the API
    base_url = os.getenv('GITHUB_BACKEND_BASED_URL')
    # Combine the base URL and API path
    full_url = f"{base_url}/api/repositories"
    print(full_url)
    # Send a GET request to the external API with the parameters
    response = requests.get(full_url, params=params)

    # If the request is successful (status code 200), return the JSON data
    if response.status_code == 200:
        return response.json()  # Return the data as JSON
    else:
        # If the request fails, return an error message
        print(f"Error: Failed to fetch data from the API. Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")

datas = fetch_repositories({'is_trending': True})
print(datas)