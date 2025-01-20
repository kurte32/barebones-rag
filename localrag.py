import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import ollama
from openai import OpenAI
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.panel import Panel
import logging
import re
import hashlib

# Initialize Rich Console for enhanced terminal output
console = Console()

# Configure logging to capture errors and important events
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR,
    encoding='utf-8'  # Ensure logs handle UTF-8 encoding
)

# Initialize embedding cache (reserved for future optimizations)
embedding_cache = {}

def compute_file_hash(filepath):
    """Computes the SHA256 hash of a given file."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        console.print(f"[red]Failed to compute hash for {filepath}: {e}. Ensure the file exists and has read permissions.[/red]")
        logging.error(f"Failed to compute hash for {filepath}: {e}")
        return None

def save_embeddings_with_hash(embeddings, filepath='embeddings.pt', vault_path='vault.txt'):
    """Saves embeddings along with the SHA256 hash of vault.txt for integrity verification."""
    try:
        vault_hash = compute_file_hash(vault_path)
        data = {
            'embeddings': embeddings.cpu().numpy(),  # Convert to NumPy array for portability
            'vault_hash': vault_hash
        }
        torch.save(data, filepath)
        console.print(f"[green]Embeddings and vault hash successfully saved to {filepath}.[/green]\nThis allows for reusing embeddings in future runs without regeneration.\n")
    except Exception as e:
        console.print(f"[red]Failed to save embeddings with hash: {e}. Check write permissions and data integrity.[/red]")
        logging.error(f"Failed to save embeddings with hash: {e}")

def load_embeddings_with_hash(filepath='embeddings.pt', vault_path='vault.txt'):
    """Loads embeddings and verifies the SHA256 hash of vault.txt to ensure data consistency."""
    if not os.path.exists(filepath):
        console.print(f"[yellow]{filepath} not found. Proceeding to generate new embeddings.\n[/yellow]")
        return torch.tensor([])
    try:
        data = torch.load(filepath, weights_only=False)  # Load the entire dictionary
        embeddings = torch.tensor(data.get('embeddings', []), dtype=torch.float32)  # Convert back to tensor
        saved_hash = data.get('vault_hash', None)
        current_hash = compute_file_hash(vault_path)
        if saved_hash != current_hash:
            console.print("[yellow]The vault content has changed since embeddings were generated. Regenerating embeddings to maintain consistency.[/yellow]\n")
            return torch.tensor([])
        console.print(f"[green]Embeddings successfully loaded from {filepath}.[/green]\nThis accelerates future queries without the need to regenerate embeddings.\n")
        return embeddings
    except Exception as e:
        console.print(f"[red]Failed to load embeddings with hash: {e}. Proceeding to generate new embeddings.\n[/red]")
        logging.error(f"Failed to load embeddings with hash: {e}")
        return torch.tensor([])

def generate_embeddings(vault_content, embedding_model, device='cpu'):
    """Generates embeddings for each document in the vault using the specified embedding model."""
    embeddings = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit embedding generation tasks for each document
        futures = {executor.submit(get_embedding, content, embedding_model): content for content in vault_content}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("Generating embeddings for each document...", total=len(futures))
            for future in as_completed(futures):
                try:
                    embedding = future.result()
                    if embedding:
                        embeddings.append(embedding)
                except Exception as e:
                    console.print(f"[red]Error generating embedding for content: {e}[/red]")
                    logging.error(f"Error generating embedding for content: {e}")
                progress.advance(task)
    if embeddings:
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        return embeddings_tensor / embeddings_tensor.norm(dim=1, keepdim=True)  # Normalize embeddings
    return torch.tensor([])

def get_embedding(content, embedding_model):
    """Retrieves the embedding for a single piece of content using the specified embedding model."""
    try:
        response = ollama.embeddings(model=embedding_model, prompt=content)
        return response.get("embedding")
    except Exception as e:
        console.print(f"[red]Failed to get embedding for content: {e}[/red]")
        logging.error(f"Failed to get embedding for content: {e}")
        return None

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, embedding_model, device='cpu', top_k=7, similarity_threshold=0.6):
    """Retrieves the most relevant contexts from the vault based on the rewritten input."""
    if vault_embeddings.nelement() == 0:
        console.print("[cyan]No embeddings loaded. Cannot retrieve relevant context.[/cyan]\n")
        return []

    try:
        # Generate embedding for the rewritten input query
        response = ollama.embeddings(model=embedding_model, prompt=rewritten_input)
        input_embedding = response.get("embedding")

        if not input_embedding:
            console.print("[red]No embedding returned for the input provided. Please verify the input and try again.[/red]\n")
            logging.error("No embedding returned for the input provided.")
            return []

        input_tensor = torch.tensor(input_embedding, dtype=torch.float32, device=device)
        input_tensor = input_tensor / input_tensor.norm()  # Normalize input embedding

        if vault_embeddings.device != device:
            vault_embeddings = vault_embeddings.to(device)

        # Ensure vault_embeddings are normalized
        vault_embeddings = vault_embeddings / vault_embeddings.norm(dim=1, keepdim=True)

        # Compute cosine similarity between input embedding and all vault embeddings
        cos_scores = torch.mm(vault_embeddings, input_tensor.unsqueeze(1)).squeeze(1)

        # Get top_k indices based on similarity scores
        topk_scores, topk_indices = torch.topk(cos_scores, k=top_k)

        # Filter out contexts below the similarity threshold
        topk_filtered_indices = [idx for idx, score in zip(topk_indices.tolist(), topk_scores.tolist()) if score >= similarity_threshold]

        # Debugging: Print similarity scores and top indices
        console.print(f"[blue]Similarity Scores: {topk_scores.tolist()}[/blue]\n")
        console.print(f"[blue]Top Indices: {topk_filtered_indices}[/blue]\n")
        console.print(f"[blue]Top Contexts: {len(topk_filtered_indices)}[/blue]\n")
        

        if not topk_filtered_indices:
            console.print("[yellow]No context meets the established similarity threshold. Consider lowering the threshold to obtain more results.[/yellow]\n")

        return [vault_content[idx].strip() for idx in topk_filtered_indices]
    except Exception as e:
        console.print(f"[red]Error in get_relevant_context: {e}[/red]\n")
        logging.error(f"Error in get_relevant_context: {e}")
        return []

def rewrite_query(user_input_json, conversation_history, llm_model, client):
    """
    Rewrites the user's query based on the conversation history to enhance clarity and specificity.

    Args:
        user_input_json (str): JSON string containing the original user query.
        conversation_history (list): List of dictionaries representing the conversation history.
        llm_model (str): The LLM model to use for rewriting.
        client (OpenAI): The initialized LLM API client.

    Returns:
        dict: A dictionary containing the rewritten query.
    """
    try:
        user_input = json.loads(user_input_json)["Query"]
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
        prompt = f"""You are an assistant tasked with rewriting user queries to enhance clarity and specificity based on the recent conversation history. 

The rewritten query should:
- Maintain the original intent and meaning.
- Incorporate relevant elements from the conversation history to provide context.
- Be clear and specific to facilitate accurate retrieval of information from the provided context.
- Avoid introducing any new topics or deviating from the original query.

DO NOT provide any answers or additional explanations. Only return the rewritten query text.

Conversation History:
{context}

Original Query: [{user_input}]

Rewritten Query:
"""
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=200,
            n=1,
            temperature=0.1,
        )
        rewritten_query = response.choices[0].message.content.strip()
        return {"Rewritten Query": rewritten_query}
    except Exception as e:
        console.print(f"[red]Error in rewrite_query: {e}[/red]\n")
        logging.error(f"Error in rewrite_query: {e}")
        return {"Rewritten Query": ""}

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, llm_model, embedding_model, conversation_history, client):
    """
    Handles the chat interaction with the LLM, including query rewriting and context retrieval.

    Args:
        user_input (str): The user's input query.
        system_message (str): The system message defining the assistant's role.
        vault_embeddings (torch.Tensor): Tensor of precomputed vault embeddings.
        vault_content (list): List of strings representing the vault content.
        llm_model (str): The LLM model to use.
        embedding_model (str): The embedding model to use.
        conversation_history (list): List of dictionaries representing the conversation history.
        client (OpenAI): The initialized LLM API client.

    Returns:
        str: The assistant's response.
    """
    # Append the user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        # Rewrite the user's query based on conversation history
        rewritten_query_data = rewrite_query(json.dumps(query_json), conversation_history, llm_model, client)
        rewritten_query = rewritten_query_data.get("Rewritten Query", "")
        if rewritten_query:
            console.print(Panel(f"Original Query: {user_input}", style="magenta"))
            console.print(Panel(f"Rewritten Query: {rewritten_query}", style="magenta"))
        else:
            rewritten_query = user_input
    else:
        rewritten_query = user_input

    # Retrieve relevant context based on the rewritten query
    relevant_context = get_relevant_context(
        rewritten_query,
        vault_embeddings,
        vault_content,
        embedding_model=embedding_model,  # Use separate embedding model
        device='cpu',  # Adjust based on your setup
        top_k=7,
        similarity_threshold=0.6  # Adjusted threshold
    )
    if relevant_context:
        context_str = "\n".join(relevant_context)
        user_input_with_context = f"{user_input}\n\nRelevant Context:\n{context_str}"
        console.print(Panel(f"Context Retrieved from Documents:\n\n{context_str}", style="cyan"))
    else:
        user_input_with_context = user_input
        console.print("[cyan]No relevant context found.[/cyan]\n")

    # Update the last user message with the context
    conversation_history[-1]["content"] = user_input_with_context

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    try:
        # Generate response from the LLM
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            max_tokens=2000,
        )
        assistant_message = response.choices[0].message.content
        # Append the assistant's response to the conversation history
        conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    except Exception as e:
        console.print(f"[red]Error generating response with the LLM: {e}[/red]\n")
        logging.error(f"Error generating response with the LLM: {e}")
        return "An error occurred while generating the response."

def main():
    """Main function to run the RAG system."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="RAG Chat - Retrieval-Augmented Generation System")
    parser.add_argument("--model", default="llama3.2", help="LLM model to use")
    parser.add_argument("--embedding-model", default="mxbai-embed-large", help="Embedding model to use")
    parser.add_argument("--device", default="cpu", help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--use-cache", action='store_true', help="Load existing embeddings if available")
    parser.add_argument("--save-embeddings", action='store_true', help="Save embeddings after generation")
    parser.add_argument("--embedding-file", default="embeddings.pt", help="Path to embeddings file")
    args = parser.parse_args()

    # Initialize LLM API client
    console.print("[bold green]Initializing the LLM API client...[/bold green]\n")
    try:
        client = OpenAI(
            base_url='http://localhost:11434/v1',  # Ensure this URL is correct for your setup
            api_key='your_api_key_here'  # Replace with your actual API key if required
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize the LLM API client: {e}[/red]\n")
        logging.error(f"Failed to initialize the LLM API client: {e}")
        return

    # Load the vault content
    console.print("[bold green]Loading vault content...[/bold green]\n")
    vault_content = []
    if os.path.exists("vault.txt"):
        try:
            with open("vault.txt", "r", encoding="utf-8") as f:
                content = f.read()
                vault_content = content.splitlines()
            console.print(f"[green]Vault content loaded from 'vault.txt'. Total documents: {len(vault_content)}.[/green]\n")
        except Exception as e:
            console.print(f"[red]Failed to read 'vault.txt': {e}[/red]\n")
            logging.error(f"Failed to read 'vault.txt': {e}")
    else:
        console.print("[yellow]File 'vault.txt' not found. Proceeding with an empty vault.[/yellow]\n")

    # Load or generate embeddings with hash verification
    vault_embeddings = torch.tensor([])
    if vault_content:
        if args.use_cache:
            vault_embeddings = load_embeddings_with_hash(filepath=args.embedding_file, vault_path='vault.txt')
        if vault_embeddings.nelement() == 0:
            console.print("[bold green]Generating new embeddings for the vault content...[/bold green]\n")
            vault_embeddings = generate_embeddings(
                vault_content,
                embedding_model=args.embedding_model,  # Use separate embedding model
                device=args.device
            )
            if args.save_embeddings:
                save_embeddings_with_hash(vault_embeddings, filepath=args.embedding_file, vault_path='vault.txt')
            if vault_embeddings.nelement() == 0:
                console.print("[yellow]No embeddings generated. Proceeding without vault context.[/yellow]\n")
            else:
                console.print(f"[green]Generated embeddings for {len(vault_embeddings)} documents.[/green]\n")
    else:
        vault_embeddings = torch.tensor([])

    # Initialize conversation history
    conversation_history = []
    system_message = (
        "You are a knowledgeable assistant specialized in extracting and summarizing information from the provided context. "
        "Respond to user queries based solely on the given context without introducing any external information. "
        "If the answer is not present in the context, clearly state that the information is not available."
    )

    # Conversation loop
    console.print("[bold green]Starting the conversation loop...[/bold green]\n")
    while True:
        try:
            user_input = Prompt.ask("[yellow]Ask a query about your documents (or type 'quit' to exit):[/yellow]\n")
            if user_input.lower() == 'quit':
                console.print("[bold red]Exiting the chat. Goodbye![/bold red]\n")
                break

            # Handle chat interaction
            response = ollama_chat(
                user_input,
                system_message,
                vault_embeddings,
                vault_content,
                args.model,
                args.embedding_model,  # Pass embedding model
                conversation_history,
                client
            )
            console.print(Panel(f"Response:\n\n{response}", style="green"))
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user. Exiting the chat...[/bold red]\n")
            break
        except Exception as e:
            console.print(f"[red]An unexpected error occurred: {e}[/red]\n")
            logging.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
