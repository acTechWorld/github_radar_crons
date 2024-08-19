from api.repositories import fetch_repositories
from api.ollama import chat_multi
from transformers import AutoTokenizer
import json

# Load a tokenizer (match the LLM used in Ollama)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def chunk_json_by_tokens(json_list, max_tokens=4096):
    """Splits a JSON list of objects into chunks of ≤ max_tokens, preserving whole objects."""
    chunks = []
    current_chunk = []
    current_tokens = 0

    for obj in json_list:
        obj_text = json.dumps(obj, ensure_ascii=False)  # Convert object to string
        token_count = len(tokenizer.tokenize(obj_text))  # Count tokens

        # If adding this object exceeds max_tokens, start a new chunk
        if current_tokens + token_count > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(obj)  # Add object to chunk
        current_tokens += token_count

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

def getTopicsTrendingRepositories():
    res = fetch_repositories({"is_trending": True})
    datas=fetch_repositories({"is_trending": True, "limit": res["totalCount"]})
    repositories = datas['items']
    def mapping(repo):
        return {
            "name": repo['name'],
            "description": repo['description']
        }
    chunks = chunk_json_by_tokens(map(mapping, repositories), max_tokens=30000)
    size= len(list(map(mapping, repositories)))
    # Add instruction to each chunk
    instruction_text = f"""
--- WHAT YOU SHOULD DO ---
You will process a list of {size} GitHub repositories in JSON format and extract only the repository names.

### INPUT FORMAT:
- Each entry in the list is a JSON object with this structure:
  {{
    "name": "[REPOSITORY_NAME]",
    "description": "[REPOSITORY_DESCRIPTION]"
  }}

### TASK:
- Extract only the `name` field from each JSON object.
- Ensure the output contains exactly {size} names (one per repository).
- Give me the result yourself and don't create a script to do it

### OUTPUT FORMAT:
- The output should contain {size} lines, one per repository.
- Each line should be numbered sequentially from 1 to {size}.
- The response should be in English only.

### EXAMPLE OUTPUT:
1. x
2. shadcn-vue
3. [REPOSITORY_NAME 3]
...
{size}. [REPOSITORY_NAME {size}]
"""
    formatted_chunks = [f"{instruction_text}\n\n" + json.dumps(chunk, indent=2, ensure_ascii=False) for chunk in chunks]

    print(formatted_chunks)
    # Send each chunk to Ollama
    for i, chunk in enumerate(formatted_chunks):
        print(f"Processing chunk {i+1}/{len(formatted_chunks)}...\n")
        
        response = chat_multi(messages=[{"role": "user", "content": chunk}])
    
    return response["message"]["content"]

res = getTopicsTrendingRepositories()
print(res)


