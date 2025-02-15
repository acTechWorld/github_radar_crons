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
    datas=fetch_repositories({"is_trending": True, "limit": res["totalCount"], "created_after": "2024-06-01"})
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

    # Instruction text for LLM to analyze all clusters at once
    instruction_text = """
--- TASK OVERVIEW ---

### INPUT FORMAT:
- You will be provided with **multiple clusters** of GitHub repositories.
- Each cluster contains repositories that share a **common theme**.
- A cluster is represented as:
  {
    "cluster_id": [Cluster_ID],  # The ID of the cluster
    "repositories": [
      {
        "name": "[REPOSITORY_NAME]",  
        "description": "[REPOSITORY_DESCRIPTION]"  
      },
      ...
    ]
  }

### YOUR TASK:
For each **cluster**, generate:
1️⃣ **A category name** that accurately represents the common theme of the repositories.  
2️⃣ **A clear and concise summary** explaining the key trends, technologies, or innovation in that category.  
3️⃣ **A list of project names** inside that cluster.  
4️⃣ **Three innovative project ideas** that could emerge from this trend.  
   - These should be **unique, specific, and practical** (avoid generic AI-powered descriptions).  

### OUTPUT FORMAT:
- Return a **JSON array only** with the following structure:
[
    {   
        "cluster_id": [Cluster_ID],
        "category": "[TREND_NAME]",  
        "summary": "[Detailed explanation of the trend]",
        "projects": ["Project 1", "Project 2", ...],
        "innovative_projects": [
            {
                "name": "[Project Idea 1]",
                "description": "[Unique and technically relevant project idea]"
            },
            {
                "name": "[Project Idea 2]",
                "description": "[Distinct from others and clearly useful]"
            },
            {
                "name": "[Project Idea 3]",
                "description": "[Solves a real-world problem effectively]"
            }
        ]
    },
    ...
]

### IMPORTANT NOTES:
- 🚀 **Be original**: Avoid overused terms like "AI-powered tool" without explaining how it actually works.  
- 🔬 **Be specific**: Instead of "Real-Time Debugging," describe a **live execution visualizer for JS frameworks**.  
- 💡 **Think beyond AI**: Some innovations could be **blockchain-based, edge computing, WebAssembly, or quantum programming**.  
- 📜 **Respond with valid JSON only**: No additional text, introductions, or summaries.
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
    print([
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": formatted_input},
        {"role": "assistant", "content": seeded_prompt}  # Enforce array format
    ])
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
