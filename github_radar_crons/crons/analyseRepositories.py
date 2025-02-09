from api.repositories import fetch_repositories
from api.ollama import chat_multi
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
import numpy as np

# Load tokenizer & embedding model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Small & fast embedding model

def getTopicsTrendingRepositories():
    # Fetch trending repositories
    res = fetch_repositories({"is_trending": True, "created_after": "2024-06-01"})
    print(res["totalCount"])
    datas = fetch_repositories({"is_trending": True, "limit": res["totalCount"], "created_after": "2024-06-01"})
    repositories = datas["items"]

    # Map repositories to relevant data
    repo_data = [{"name": repo["name"], "description": repo["description"]} for repo in repositories]

    # Generate embeddings
    descriptions = [repo["description"] for repo in repo_data if repo["description"]]
    embeddings = embedding_model.encode(descriptions)

    # Cluster repositories into topics
    num_clusters = min(10, len(descriptions))  # Limit to 10 clusters or total projects
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    # Organize repositories into clusters
    clustered_repos = {i: [] for i in range(num_clusters)}
    for i in range(min(len(repo_data), len(clusters))):
        clustered_repos[clusters[i]].append(repo_data[i])

    # Instruction text for LLM to analyze all clusters at once and suggest new projects
    instruction_text = """
--- WHAT YOU SHOULD DO ---

### INPUT FORMAT:
- You will be provided with multiple groups (clusters) of GitHub repositories.
- Each cluster is a JSON object with the following structure:
  {
    "cluster_id": [Cluster_ID],  # The ID of the cluster
    "repositories": [
      {
        "name": "[REPOSITORY_NAME]",  # Name of the repository
        "description": "[REPOSITORY_DESCRIPTION]"  # Description of what the repository does
      },
      ...
    ]
  }
- Each cluster represents related repositories sharing a common trend or topic.

### TASK:
- Analyze **each cluster individually** and provide:
  1. **A category name** that describes the overall theme of the cluster.
  2. **A short summary** explaining the repositories in that cluster.
  3. **A list of project names** in the cluster.
  4. **Three innovative project ideas** related to the cluster theme.

### OUTPUT FORMAT:
- Your response must be **only a JSON array** with the following structure:
[
    {   
        "cluster_id": [Cluster_ID],
        "category": "[TREND_NAME]",  # A meaningful label for the trend
        "summary": "[Short description of the trend or technology]",
        "projects": ["Project 1", "Project 2", ...],  # Existing projects in this trend
        "innovative_projects": [
            {
                "name": "[Innovative Project 1]",
                "description": "[Brief description of why it's innovative]"
            },
            {
                "name": "[Innovative Project 2]",
                "description": "[Brief description of why it's innovative]"
            },
            {
                "name": "[Innovative Project 3]",
                "description": "[Brief description of why it's innovative]"
            }
        ]
    },
    ...
]
- **Do not add any extra text outside of the JSON array.**
- **Respond only with valid JSON. Do not write an introduction or summary.**
"""

    # Combine all clustered repositories into a list for LLM input
    all_clusters = []
    for cluster_id, cluster_repos in clustered_repos.items():
        all_clusters.append({
            "cluster_id": cluster_id,
            "repositories": cluster_repos
        })

    # Convert the list into a JSON string
    formatted_input = json.dumps(all_clusters, indent=2, ensure_ascii=False)

    # Seed the LLM response with an opening bracket '[' to enforce JSON array output
    seeded_prompt = "[\n"

    # Send all clusters to LLM with a **forced prefix**
    response = chat_multi([
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": formatted_input},
        {"role": "assistant", "content": seeded_prompt}  # Enforce array format
    ])

    response_content = response['message']['content'].strip()

    # Ensure response starts with '[' and ends with ']' (fallback if needed)
    if not response_content.startswith("["):
        response_content = "[" + response_content
    if not response_content.endswith("]"):
        response_content += "]"

    # Try parsing the response into JSON
    try:
        analyzed_clusters = json.loads(response_content)
    except json.JSONDecodeError:
        print("⚠️ Error: LLM response is not valid JSON. Saving raw response instead.")
        analyzed_clusters = response_content  # Save raw response for debugging

    # Save the final result to a single JSON file
    output_filename = "all_trending_repositories_analysis.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(analyzed_clusters, f, ensure_ascii=False, indent=2)

    print(f"✅ Analysis saved to {output_filename}")
    return analyzed_clusters

# Run the function
res = getTopicsTrendingRepositories()
print(res)
