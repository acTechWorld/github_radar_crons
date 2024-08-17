from api.repositories import fetch_repositories
from api.ollama import chat

def getTopicsTrendingRepositories():
    res = fetch_repositories({"is_trending": True})
    datas=fetch_repositories({"is_trending": True, "limit": res["totalCount"]})
    repositories = datas['items']
    def mapping(repo):
        return {
            "name": repo['name'],
            "description": repo['description']
        }
    mappedDatas= map(mapping, repositories)
    prompt= f"Here is a list of Github repositories: {list(mappedDatas)}. Analyse the list and merge them into a list of 10 project ideas that are the most represented"
    #resOllama= chat(prompt)
    return resOllama

res = getTopicsTrendingRepositories()
print(res['message'])