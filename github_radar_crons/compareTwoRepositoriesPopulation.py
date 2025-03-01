from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from api.ollama import chat_multi
import json
from api.repositories import fetch_repositories, saveAiReposComparaisons, getAiRepoAnalysisByNameAndOwner, saveAiRepoAnalysis

model = SentenceTransformer('all-MiniLM-L6-v2')


# Function to compute similarity score
def similarity_score(a, b):
    vec_a = model.encode([a], convert_to_tensor=True)
    vec_b = model.encode([b], convert_to_tensor=True)
    vec_a = vec_a.cpu().numpy().reshape(1, -1)  # Reshape to 2D array (1, n_features)
    vec_b = vec_b.cpu().numpy().reshape(1, -1)  # Reshape to 2D array (1, n_features)
    
    return cosine_similarity(vec_a, vec_b)[0][0]

# Function to summarize a project's README content
def get_repo_summary_from_ai(readme):
    instruction_text = "Summarize the project's main idea and the problem it solves."
    instruction_text = """
### TASK OVERVIEW ###

#### INPUT FORMAT:
- You will be provided with the text of a GitHub README file.

#### YOUR TASK:
1. **Translation**: If the README content is not entirely in English, translate the entire content into English first.
2. **Summarization**: From the (translated) README, extract:
   - **Main Idea** – The primary concept or purpose of the project.
   - **Problem Solved** – The specific issue or need the project addresses.

#### OUTPUT FORMAT:
- Return a **valid JSON object only**, structured as follows:

```json
{
    "main_idea": "[The project's main idea]",
    "solving": "[The problem the project solves]"
}
IMPORTANT NOTES: 
📜 Strictly return JSON – No extra text, explanations, or introductions.
🌍 All extracted content must be in English.
⚠️ If information is missing, infer from the README or return an empty string.
🔄 Ensure valid JSON formatting, including proper character escaping.
"""
    formatted_input = readme
    seeded_prompt = "{" # You can add any pre-seeding context here if needed
    
    response = chat_multi([
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": formatted_input},
        {"role": "assistant", "content": seeded_prompt}
    ])
    response_content = response['message']['content'].strip()

    # Ensure response starts with '[' and ends with ']' (fallback if needed)
    if not response_content.startswith("{"):
        response_content = "{" + response_content
    if not response_content.endswith("}"):
        response_content += "}"
    return json.loads(response_content)

def get_repo_summary(repo):
    apiRes = getAiRepoAnalysisByNameAndOwner(repo['name'], repo['owner_name'])
    if apiRes: 
        return json.loads(apiRes['content'])
    else:
        repo_summary = get_repo_summary_from_ai(repo['readme'])
        saveAiRepoAnalysis({
            "repo_name": repo['name'],
            "repo_owner": repo['owner_name'],
            "content": json.dumps(repo_summary)
        })
        return repo_summary

# Compare two repositories: React vs Vue
def compareTwoRepositoriesPopulation(list_compared, list_to_compare_with): 
    result= []
    # Fetch and summarize README content for React projects
    list_to_compare_with_summaries = {}
    for repo in list_to_compare_with:
        repo_summary = get_repo_summary(repo)
        print(repo_summary)
        list_to_compare_with_summaries[repo['name'] + '_' + repo['owner_name']] = repo_summary['main_idea'] + ' ' +  repo_summary['solving'] # Summarize the README

    # Fetch and summarize README content for Vue projects
    list_compared_summaries = {}
    for repo in list_compared:
        repo_summary = get_repo_summary(repo)
        print(repo_summary)
        list_compared_summaries[repo['name'] + '_' + repo['owner_name']] = repo_summary['main_idea'] + ' ' +  repo_summary['solving'] # Summarize the README

    for repo_name, r_summary in list_to_compare_with_summaries.items():
        scores = [(v_name, similarity_score(r_summary, v_summary)) for v_name, v_summary in list_compared_summaries.items()]
        if scores:
            closest_match = max(scores, key=lambda x: x[1])
            result.append({
                "repo_compared": repo_name,
                "similar_repo": closest_match[0],
                "value": closest_match[1]
            })
        else:
            print(f"- {repo_name} (No repositories available for comparison)")
    return result

trendingTypes = ["last_6_months"]
for trendingType in trendingTypes:
    reactRepos = fetch_repositories({"languages": "React", "languagesOperation": "OR", "hasReadMe": "true", "trendingTypes": trendingType, "trendingTypesOperation": "OR"})
    vueRepos = fetch_repositories({"languages": "Vue", "languagesOperation": "OR", "hasReadMe": "true", "trendingTypes": trendingType, "trendingTypesOperation": "OR"})

    reactRepos = [{"name": repo["name"], "owner_name": repo['owner_name'], "description": repo["description"], "readme":  repo["readme_content"]} for repo in reactRepos["items"]]
    vueRepos = [{"name": repo["name"], "owner_name": repo['owner_name'], "description": repo["description"], "readme":  repo["readme_content"]} for repo in vueRepos["items"]]
    res = compareTwoRepositoriesPopulation(vueRepos, reactRepos)
    print(res)
    saveAiReposComparaisons({"name": f"compareTwoRepositoriesPopulation_{trendingType}", "type": "compareTwoRepositoriesPopulation", "content": json.dumps(res, default=float)})
    
