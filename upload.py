import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import PyPDF2
import re
import json
import threading

# Helper function to split text into chunks
def split_text_into_chunks(text, max_length=1000):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < max_length:
            current_chunk += (sentence + " ").strip()
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Helper function to append chunks to vault.txt
def append_chunks_to_vault(chunks):
    with open("vault.txt", "a", encoding="utf-8") as vault_file:
        for chunk in chunks:
            vault_file.write(chunk.strip() + "\n")

# Function to process PDF files
def convert_pdf_to_text(progress_var):
    try:
        file_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not file_path:
            return

        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            text = ''

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + " "
                progress_var.set((page_num + 1) / num_pages * 100)
            
            text = re.sub(r'\s+', ' ', text).strip()
            chunks = split_text_into_chunks(text)
            append_chunks_to_vault(chunks)

        messagebox.showinfo("Success", "PDF content appended to vault.txt successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while processing the PDF:\n{e}")
    finally:
        progress_var.set(0)

# Function to process Text files
def upload_txtfile(progress_var):
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if not file_path:
            return

        with open(file_path, 'r', encoding="utf-8") as txt_file:
            text = txt_file.read()
        
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = split_text_into_chunks(text)
        append_chunks_to_vault(chunks)

        messagebox.showinfo("Success", "Text file content appended to vault.txt successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while processing the text file:\n{e}")
    finally:
        progress_var.set(0)

# Function to process JSON files
def upload_jsonfile(progress_var):
    try:
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if not file_path:
            return

        with open(file_path, 'r', encoding="utf-8") as json_file:
            data = json.load(json_file)
        
        text = json.dumps(data, ensure_ascii=False)
        text = re.sub(r'\s+', ' ', text).strip()
        chunks = split_text_into_chunks(text)
        append_chunks_to_vault(chunks)

        messagebox.showinfo("Success", "JSON file content appended to vault.txt successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while processing the JSON file:\n{e}")
    finally:
        progress_var.set(0)

# Function to handle PDF upload with threading
def handle_pdf_upload(progress_var):
    threading.Thread(target=convert_pdf_to_text, args=(progress_var,)).start()

# Function to handle Text upload with threading
def handle_txt_upload(progress_var):
    threading.Thread(target=upload_txtfile, args=(progress_var,)).start()

# Function to handle JSON upload with threading
def handle_json_upload(progress_var):
    threading.Thread(target=upload_jsonfile, args=(progress_var,)).start()

# Create the main window
root = tk.Tk()
root.title("Upload Files to Vault")
root.geometry("400x300")
root.resizable(False, False)

# Style configuration
style = ttk.Style(root)
style.theme_use('clam')

# Create a frame for buttons
button_frame = ttk.Frame(root, padding=20)
button_frame.pack(expand=True)

# Create a progress bar
progress_var = tk.DoubleVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(fill='x', padx=20, pady=(0, 20))

# Create buttons with better styling
pdf_button = ttk.Button(button_frame, text="Upload PDF", command=lambda: handle_pdf_upload(progress_var))
pdf_button.pack(fill='x', pady=5)

txt_button = ttk.Button(button_frame, text="Upload Text File", command=lambda: handle_txt_upload(progress_var))
txt_button.pack(fill='x', pady=5)

json_button = ttk.Button(button_frame, text="Upload JSON File", command=lambda: handle_json_upload(progress_var))
json_button.pack(fill='x', pady=5)

# Run the main event loop
root.mainloop()
