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

# Inicializar la Consola de Rich
console = Console()

# Configurar el registro de logs
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

# Inicializar la caché de embeddings (si es necesario en el futuro)
embedding_cache = {}

def compute_file_hash(filepath):
    """Calcula el hash SHA256 de un archivo."""
    sha256 = hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        console.print(f"[red]Error al calcular el hash para el archivo {filepath}: {e}. Asegúrese de que el archivo exista y tenga permisos de lectura adecuados.[/red]")
        logging.error(f"Error al calcular el hash para el archivo {filepath}: {e}")
        return None

def save_embeddings_with_hash(embeddings, filepath='embeddings.pt', vault_path='vault.txt'):
    """Guarda los embeddings junto con el hash de vault.txt."""
    try:
        vault_hash = compute_file_hash(vault_path)
        data = {
            'embeddings': embeddings.cpu().numpy(),  # Convertir a numpy para portabilidad
            'vault_hash': vault_hash
        }
        torch.save(data, filepath)
        console.print(f"[green]Embeddings y el hash de la bóveda se han guardado exitosamente en {filepath}.[/green]\nEsto permitirá reutilizar los embeddings en futuras ejecuciones sin necesidad de regenerarlos.\n")
    except Exception as e:
        console.print(f"[red]Error al guardar los embeddings con hash: {e}. Verifique los permisos de escritura y la integridad de los datos.[/red]")
        logging.error(f"Error al guardar los embeddings con hash: {e}")

def load_embeddings_with_hash(filepath='embeddings.pt', vault_path='vault.txt'):
    """Carga los embeddings y verifica el hash de vault.txt."""
    if not os.path.exists(filepath):
        console.print(f"[yellow]El archivo {filepath} no se encontró. Procediendo a generar nuevos embeddings.\n[/yellow]")
        return torch.tensor([])
    try:
        data = torch.load(filepath, weights_only=False)  # Cambiado a False para cargar el diccionario completo
        embeddings = torch.tensor(data.get('embeddings', []), dtype=torch.float32)  # Convertir de nuevo a tensor
        saved_hash = data.get('vault_hash', None)
        current_hash = compute_file_hash(vault_path)
        if saved_hash != current_hash:
            console.print("[yellow]El contenido de la bóveda ha cambiado desde que se generaron los embeddings. Procediendo a regenerarlos para mantener la coherencia.[/yellow]\n")
            return torch.tensor([])
        console.print(f"[green]Los embeddings se han cargado exitosamente desde {filepath}.[/green]\nEsto permite acelerar las consultas futuras sin necesidad de regenerar embeddings.\n")
        return embeddings
    except Exception as e:
        console.print(f"[red]Error al cargar los embeddings con hash: {e}. Procediendo a generar nuevos embeddings.\n[/red]")
        logging.error(f"Error al cargar los embeddings con hash: {e}")
        return torch.tensor([])

def generate_embeddings(vault_content, embedding_model, device='cpu'):
    """Genera embeddings para el contenido de la bóveda."""
    embeddings = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_embedding, content, embedding_model): content for content in vault_content}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("Generando embeddings para cada documento...", total=len(futures))
            for future in as_completed(futures):
                try:
                    embedding = future.result()
                    if embedding:
                        embeddings.append(embedding)
                except Exception as e:
                    console.print(f"[red]Error al generar el embedding para el contenido: {e}[/red]")
                    logging.error(f"Error al generar el embedding para el contenido: {e}")
                progress.advance(task)
    if embeddings:
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=device)
        return embeddings_tensor / embeddings_tensor.norm(dim=1, keepdim=True)  # Normalizar
    return torch.tensor([])

def get_embedding(content, embedding_model):
    """Obtiene el embedding para un contenido específico."""
    try:
        response = ollama.embeddings(model=embedding_model, prompt=content)
        return response.get("embedding")
    except Exception as e:
        console.print(f"[red]Error al obtener el embedding para el contenido: {e}[/red]")
        logging.error(f"Error al obtener el embedding para el contenido: {e}")
        return None

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, embedding_model, device='cpu', top_k=7, similarity_threshold=0.6):
    """Recupera los contextos más relevantes de la bóveda."""
    if vault_embeddings.nelement() == 0:
        console.print("[cyan]No se encontraron embeddings cargados. No se puede recuperar contexto relevante.[/cyan]\n")
        return []

    try:
        response = ollama.embeddings(model=embedding_model, prompt=rewritten_input)
        input_embedding = response.get("embedding")

        if not input_embedding:
            console.print("[red]No se devolvió ningún embedding para la entrada proporcionada. Verifique la entrada y vuelva a intentarlo.[/red]\n")
            logging.error("No se devolvió ningún embedding para la entrada proporcionada.")
            return []

        input_tensor = torch.tensor(input_embedding, dtype=torch.float32, device=device)
        input_tensor = input_tensor / input_tensor.norm()  # Normalizar

        if vault_embeddings.device != device:
            vault_embeddings = vault_embeddings.to(device)

        # Asegurar que los embeddings de la bóveda están normalizados
        vault_embeddings = vault_embeddings / vault_embeddings.norm(dim=1, keepdim=True)

        # Calcular la similitud coseno
        cos_scores = torch.mm(vault_embeddings, input_tensor.unsqueeze(1)).squeeze(1)

        # Obtener los índices top_k basados en las puntuaciones de similitud
        topk_scores, topk_indices = torch.topk(cos_scores, k=top_k)

        # Filtrar los contextos por encima del umbral de similitud
        topk_filtered_indices = [idx for idx, score in zip(topk_indices.tolist(), topk_scores.tolist()) if score >= similarity_threshold]

        # Depuración: Imprimir puntuaciones de similitud y índices
        console.print(f"[blue]Puntuaciones de Similitud: {topk_scores.tolist()}[/blue]\n")
        console.print(f"[blue]Índices de los Contextos Top: {topk_filtered_indices}[/blue]\n")

        if not topk_filtered_indices:
            console.print("[yellow]Ningún contexto cumple con el umbral de similitud establecido. Considere reducir el umbral para obtener más resultados.[/yellow]\n")

        return [vault_content[idx].strip() for idx in topk_filtered_indices]
    except Exception as e:
        console.print(f"[red]Error al recuperar contexto relevante: {e}[/red]\n")
        logging.error(f"Error al recuperar contexto relevante: {e}")
        return []

def rewrite_query(user_input_json, conversation_history, llm_model, client):
    """
    Reescribe la consulta del usuario basada en el historial de la conversación para mejorar la claridad y especificidad.

    Args:
        user_input_json (str): Cadena JSON que contiene la consulta original del usuario.
        conversation_history (list): Lista de diccionarios que representan el historial de la conversación.
        llm_model (str): El modelo LLM a utilizar para la reescritura.
        client (OpenAI): El cliente API LLM inicializado.

    Returns:
        dict: Un diccionario que contiene la consulta reescrita.
    """
    try:
        user_input = json.loads(user_input_json)["Query"]
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
        prompt = f"""Eres un asistente encargado de reescribir las consultas de los usuarios para mejorar la claridad y especificidad basándote en el historial reciente de la conversación.

La consulta reescrita debe:
- Mantener la intención y significado original.
- Incorporar elementos relevantes del historial de la conversación para proporcionar contexto.
- Ser clara y específica para facilitar la recuperación precisa de información del contexto proporcionado.
- Evitar introducir nuevos temas o desviarse de la consulta original.

NO proporciones respuestas ni explicaciones adicionales. Solo devuelve el texto de la consulta reescrita.

Historial de la Conversación:
{context}

Consulta Original: [{user_input}]

Consulta Reescrita:
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
        console.print(f"[red]Error al reescribir la consulta: {e}[/red]\n")
        logging.error(f"Error al reescribir la consulta: {e}")
        return {"Rewritten Query": ""}

def ollama_chat(user_input, system_message, vault_embeddings, vault_content, llm_model, embedding_model, conversation_history, client):
    """
    Maneja la interacción de chat con el LLM, incluyendo la reescritura de consultas y la recuperación de contexto.

    Args:
        user_input (str): La consulta de entrada del usuario.
        system_message (str): El mensaje del sistema que define el rol del asistente.
        vault_embeddings (torch.Tensor): Tensor de embeddings precomputados de la bóveda.
        vault_content (list): Lista de cadenas que representan el contenido de la bóveda.
        llm_model (str): El modelo LLM a utilizar.
        embedding_model (str): El modelo de embeddings a utilizar.
        conversation_history (list): Lista de diccionarios que representan el historial de la conversación.
        client (OpenAI): El cliente API LLM inicializado.

    Returns:
        str: La respuesta del asistente.
    """
    conversation_history.append({"role": "user", "content": user_input})

    if len(conversation_history) > 1:
        query_json = {
            "Query": user_input,
            "Rewritten Query": ""
        }
        rewritten_query_data = rewrite_query(json.dumps(query_json), conversation_history, llm_model, client)
        rewritten_query = rewritten_query_data.get("Rewritten Query", "")
        if rewritten_query:
            console.print(Panel(f"Consulta Original: {user_input}", style="magenta"))
            console.print(Panel(f"Consulta Reescrita: {rewritten_query}", style="magenta"))
        else:
            rewritten_query = user_input
    else:
        rewritten_query = user_input

    relevant_context = get_relevant_context(
        rewritten_query,
        vault_embeddings,
        vault_content,
        embedding_model=embedding_model,  # Usar modelo de embeddings separado
        device='cpu',  # Ajustar según tu configuración
        top_k=3,
        similarity_threshold=0.6  # Umbral ajustado
    )
    if relevant_context:
        context_str = "\n".join(relevant_context)
        user_input_with_context = f"{user_input}\n\nContexto Relevante:\n{context_str}"
        console.print(Panel(f"Contexto Recuperado de los Documentos:\n\n{context_str}", style="cyan"))
    else:
        user_input_with_context = user_input
        console.print("[cyan]No se encontró contexto relevante.[/cyan]\n")

    conversation_history[-1]["content"] = user_input_with_context

    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            max_tokens=2000,
        )
        assistant_message = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_message})
        return assistant_message
    except Exception as e:
        console.print(f"[red]Error al generar la respuesta con el LLM: {e}[/red]\n")
        logging.error(f"Error al generar la respuesta con el LLM: {e}")
        return "Ocurrió un error al generar la respuesta."

def main():
    """Función principal para ejecutar el sistema RAG."""
    # Analizar los argumentos de la línea de comandos
    parser = argparse.ArgumentParser(description="RAG Chat - Sistema de Recuperación y Generación de Información")
    parser.add_argument("--model", default="llama3.2", help="Modelo LLM a utilizar")
    parser.add_argument("--embedding-model", default="mxbai-embed-large", help="Modelo de embeddings a utilizar")
    parser.add_argument("--device", default="cpu", help="Dispositivo a utilizar ('cpu' o 'cuda')")
    parser.add_argument("--use-cache", action='store_true', help="Cargar embeddings existentes si están disponibles")
    parser.add_argument("--save-embeddings", action='store_true', help="Guardar embeddings después de la generación")
    parser.add_argument("--embedding-file", default="embeddings.pt", help="Ruta al archivo de embeddings")
    args = parser.parse_args()

    # Inicializar el cliente API del LLM
    console.print("[bold green]Inicializando el cliente API del LLM...[/bold green]\n")
    try:
        client = OpenAI(
            base_url='http://localhost:11434/v1',  # Asegúrate de que esta URL es correcta
            api_key='tu_api_key_aquí'  # Reemplaza con tu clave API real si es necesario
        )
    except Exception as e:
        console.print(f"[red]Error al inicializar el cliente API del LLM: {e}[/red]\n")
        logging.error(f"Error al inicializar el cliente API del LLM: {e}")
        return

    # Cargar el contenido de la bóveda
    console.print("[bold green]Cargando el contenido de la bóveda...[/bold green]\n")
    vault_content = []
    if os.path.exists("vault.txt"):
        try:
            with open("vault.txt", "r", encoding="utf-8") as f:
                content = f.read()
                vault_content = content.splitlines()
            console.print(f"[green]Se ha cargado el contenido de la bóveda desde 'vault.txt'. Total de documentos: {len(vault_content)}.[/green]\n")
        except Exception as e:
            console.print(f"[red]Error al leer 'vault.txt': {e}[/red]\n")
            logging.error(f"Error al leer 'vault.txt': {e}")
    else:
        console.print("[yellow]El archivo 'vault.txt' no se encontró. Procediendo con una bóveda vacía.[/yellow]\n")

    # Cargar o generar embeddings con verificación de hash
    vault_embeddings = torch.tensor([])
    if vault_content:
        if args.use_cache:
            vault_embeddings = load_embeddings_with_hash(filepath=args.embedding_file, vault_path='vault.txt')
        if vault_embeddings.nelement() == 0:
            console.print("[bold green]Generando nuevos embeddings para el contenido de la bóveda...[/bold green]\n")
            vault_embeddings = generate_embeddings(
                vault_content,
                embedding_model=args.embedding_model,  # Usar modelo de embeddings separado
                device=args.device
            )
            if args.save_embeddings:
                save_embeddings_with_hash(vault_embeddings, filepath=args.embedding_file, vault_path='vault.txt')
            if vault_embeddings.nelement() == 0:
                console.print("[yellow]No se generaron embeddings. Procediendo sin contexto de la bóveda.[/yellow]\n")
            else:
                console.print(f"[green]Se han generado embeddings para {len(vault_embeddings)} documentos.[/green]\n")
    else:
        vault_embeddings = torch.tensor([])

    # Inicializar el historial de la conversación
    conversation_history = []
    system_message = (
        "Eres un asistente experto en extraer y resumir información del contexto proporcionado. "
        "Responde a las consultas de los usuarios basándote únicamente en el contexto dado sin introducir información externa. "
        "Si la respuesta no está presente en el contexto, indica claramente que la información no está disponible."
    )

    # Bucle de conversación
    console.print("[bold green]Iniciando el bucle de conversación...[/bold green]\n")
    while True:
        try:
            user_input = Prompt.ask("[yellow]Realiza una consulta sobre tus documentos (o escribe 'quit' para salir):[/yellow]\n")
            if user_input.lower() == 'quit':
                console.print("[bold red]Saliendo del chat. ¡Hasta luego![/bold red]\n")
                break

            # Manejar la interacción de chat
            response = ollama_chat(
                user_input,
                system_message,
                vault_embeddings,
                vault_content,
                args.model,
                args.embedding_model,  # Pasar el modelo de embeddings
                conversation_history,
                client
            )
            console.print(Panel(f"Respuesta:\n\n{response}", style="green"))
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupción por el usuario. Saliendo del chat...[/bold red]\n")
            break
        except Exception as e:
            console.print(f"[red]Ocurrió un error inesperado: {e}[/red]\n")
            logging.error(f"Error inesperado: {e}")

if __name__ == "__main__":
    main()
