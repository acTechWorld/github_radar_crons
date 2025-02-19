import ollama
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
def split_into_sublists_max_tokens(text_list, max_tokens=1000):
    """
    Splits a list of text elements into sublists, ensuring each sublist
    stays within the max token limit based on LLaMA 3.1 tokenization.

    :param text_list: List of text elements (strings)
    :param model: The Ollama model to use for token counting (default: 'llama3')
    :param max_tokens: Max tokens per sublist
    :return: List of sublists
    """
    sublists = []
    current_sublist = []
    current_token_count = 0

    for text in text_list:
        token_count = len(ollama.tokenize(os.getenv("OLLAMA_MODEL"), text))  # Get actual token count
        
        # If adding this text exceeds max_tokens, start a new sublist
        if current_token_count + token_count > max_tokens:
            if current_sublist:
                sublists.append(current_sublist)
            current_sublist = [text]
            current_token_count = token_count
        else:
            current_sublist.append(text)
            current_token_count += token_count

    # Add the last sublist if not empty
    if current_sublist:
        sublists.append(current_sublist)

    return sublists