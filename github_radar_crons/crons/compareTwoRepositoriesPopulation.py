from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from api.ollama import chat_multi
import json
from api.repositories import fetch_repositories

model = SentenceTransformer('all-MiniLM-L6-v2')


# Function to compute similarity score
def similarity_score(a, b):
    vec_a = model.encode([a], convert_to_tensor=True)
    vec_b = model.encode([b], convert_to_tensor=True)
    vec_a = vec_a.cpu().numpy().reshape(1, -1)  # Reshape to 2D array (1, n_features)
    vec_b = vec_b.cpu().numpy().reshape(1, -1)  # Reshape to 2D array (1, n_features)
    
    return cosine_similarity(vec_a, vec_b)[0][0]

# Function to summarize a project's README content
def get_project_summary(description):
    instruction_text = "Summarize the project's main idea and the problem it solves."
    instruction_text = """
--- TASK OVERVIEW ---

### INPUT FORMAT:
- You will be provided with the text of a github readme project.

### YOUR TASK:
Summarize the readme to get:
1) The project main idea 
2) What is the problem the project solves. 

### OUTPUT FORMAT:
- Return a **JSON object only** with the following structure:
{   
    "main_idea": "[The project main idea]",  
    "solving": "[What is the problem the project solves]"
}


### IMPORTANT NOTES:
- 📜 **Respond with valid JSON only**: No additional text, introductions, or summaries.
"""
    formatted_input = description
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
    print(response_content)
    return json.loads(response_content)

# Compare two repositories: React vs Vue
def compareTwoRepositoriesPopulation(list_compared, list_to_compare_with): 

    # Fetch and summarize README content for React projects
    list_to_compare_with_summaries = {}
    for project in list_to_compare_with:
        content = get_project_summary(project['readme'])
        list_to_compare_with_summaries[project['name']] = content['main_idea'] + ' ' +  content['solving'] # Summarize the README

    # Fetch and summarize README content for Vue projects
    list_compared_summaries = {}
    for project in list_compared:
        content = get_project_summary(project['readme'])
        list_compared_summaries[project['name']] = content['main_idea'] + ' ' +  content['solving'] # Summarize the README

    for r_name, r_summary in list_to_compare_with_summaries.items():
        scores = [(v_name, similarity_score(r_summary, v_summary)) for v_name, v_summary in list_compared_summaries.items()]
        if scores:
            closest = max(scores, key=lambda x: x[1])
            print(f"- {r_name} (Closest Vue project: {closest[0]}, Similarity: {closest[1]:.2f})")
        else:
            print(f"- {r_name} (No Vue projects available for comparison)")

# Run the comparison function

reactRepos = fetch_repositories({"languages": "React", "languagesOperation": "OR", "hasReadMe": "true", "trendingTypes": "global", "trendingTypesOperation": "OR"})
vueRepos = fetch_repositories({"languages": "Vue", "languagesOperation": "OR", "hasReadMe": "true", "trendingTypes": "global", "trendingTypesOperation": "OR"})

reactRepos = [{"name": repo["name"], "description": repo["description"], "readme":  repo["readme_content"]} for repo in reactRepos["items"]]
vueRepos = [{"name": repo["name"], "description": repo["description"], "readme":  repo["readme_content"]} for repo in vueRepos["items"]]
compareTwoRepositoriesPopulation(vueRepos, reactRepos)
