import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
base_url = os.getenv('GITHUB_BACKEND_BASED_URL')

def send_request(method, path, json_body=None, params=None):
    full_url = f"{base_url}{path}"
    response = requests.request(method, full_url, json=json_body, params=params)
    
    if response.status_code in [200, 201]:
        return response.json()
    elif response.status_code == 404:
        return None
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def fetch_repositories(params):
    return send_request("GET", "/api/repositories", params=params)

def getAiReposComparaisonsByName(name):
    return send_request("GET", f"/api/aiReposComparaisons/name/{name}")

def saveAiReposComparaisons(jsonBody):
    existing_data = getAiReposComparaisonsByName(jsonBody.get("name"))
    method = "PUT" if existing_data else "POST"
    return send_request(method, "/api/aiReposComparaisons", json_body=jsonBody)

def getAiRepoAnalysisByNameAndOwner(name, owner):
    return send_request("GET", f"/api/aiRepoAnalysis/name/{name}/owner/{owner}")

def saveAiRepoAnalysis(jsonBody):
    existing_data = getAiRepoAnalysisByNameAndOwner(jsonBody.get("name"), jsonBody.get("owner"))
    method = "PUT" if existing_data else "POST"
    return send_request(method, "/api/aiRepoAnalysis", json_body=jsonBody)
