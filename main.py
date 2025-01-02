#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style

# Initialize Rich console
console = Console()

# --------------------------------------------------------------------------------
# 1. Configure OpenAI client and load environment variables
# --------------------------------------------------------------------------------
load_dotenv()  # Load environment variables from .env file
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)  # Configure for DeepSeek API

# --------------------------------------------------------------------------------
# 2. Define our schema using Pydantic for type safety
# --------------------------------------------------------------------------------
class FileToCreate(BaseModel):
    path: str
    content: str

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

class AssistantResponse(BaseModel):
    assistant_reply: str
    files_to_create: Optional[List[FileToCreate]] = None
    files_to_edit: Optional[List[FileToEdit]] = None

# --------------------------------------------------------------------------------
# 3. System prompt
# --------------------------------------------------------------------------------
system_PROMPT = dedent("""\
    You are an elite software engineer called DeepSeek Engineer with decades of experience across all programming domains.
    Your expertise spans system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning.

    Core capabilities:
    1. Code Analysis & Discussion
    2. File Operations:
       a) Read existing files
       b) Create new files
       c) Edit existing files

    Output Format:
    You must provide responses in this JSON structure:
    {
      "assistant_reply": "Your main explanation or response",
      "files_to_create": [
        {
          "path": "path/to/new/file",
          "content": "complete file content"
        }
      ],
      "files_to_edit": [
        {
          "path": "path/to/existing/file",
          "original_snippet": "exact code to be replaced",
          "new_snippet": "new code to insert"
        }
      ]
    }
""")

# --------------------------------------------------------------------------------
# 4. Helper functions
# --------------------------------------------------------------------------------

# File history for undo functionality
file_history = []

def read_local_file(file_path: str) -> str:
    """Return the text content of a local file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def create_file(path: str, content: str):
    """Create (or overwrite) a file at 'path' with the given 'content'."""
    file_path = Path(path)
    if file_path.exists():
        # Backup existing file
        with open(file_path, "r", encoding="utf-8") as f:
            backup_content = f.read()
        file_history.append(("edit", str(file_path), backup_content))
    else:
        # Mark as new file
        file_history.append(("create", str(file_path), None))
    
    # Proceed with creation
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[green]✓[/green] Created/updated file at '[cyan]{file_path}[/cyan]'")

def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    """Show a table of proposed edits."""
    if not files_to_edit:
        return
    
    table = Table(title="Proposed Edits", show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("File Path", style="cyan")
    table.add_column("Original", style="red")
    table.add_column("New", style="green")

    for edit in files_to_edit:
        table.add_row(edit.path, edit.original_snippet, edit.new_snippet)
    
    console.print(table)

def apply_diff_edit(path: str, original_snippet: str, new_snippet: str):
    """Apply a diff edit to a file."""
    try:
        content = read_local_file(path)
        if original_snippet in content:
            # Backup before editing
            file_history.append(("edit", path, content))
            updated_content = content.replace(original_snippet, new_snippet, 1)
            create_file(path, updated_content)
            console.print(f"[green]✓[/green] Applied diff edit to '[cyan]{path}[/cyan]'")
        else:
            console.print(f"[yellow]⚠[/yellow] Original snippet not found in '[cyan]{path}[/cyan]'. No changes made.", style="yellow")
    except FileNotFoundError:
        console.print(f"[red]✗[/red] File not found for diff editing: '[cyan]{path}[/cyan]'", style="red")

def undo_last_change(num_undos: int = 1):
    """Undo the last file change(s)."""
    if not file_history:
        console.print("[yellow]ℹ[/yellow] No changes to undo.", style="yellow")
        return
    
    for _ in range(num_undos):
        if not file_history:
            console.print("[yellow]ℹ[/yellow] No more changes to undo.", style="yellow")
            break
        
        last_action = file_history[-1]  # Peek at last action without removing yet
        action_type, path, backup_content = last_action
        
        try:
            if action_type == "create":
                Path(path).unlink()
                console.print(f"[green]✓[/green] Deleted file '[cyan]{path}[/cyan]' (undo creation)")
            elif action_type == "edit":
                create_file(path, backup_content)
                console.print(f"[green]✓[/green] Restored file '[cyan]{path}[/cyan]' (undo edit)")
            
            file_history.pop()  # Only remove from history if successful
        except Exception as e:
            console.print(f"[red]✗[/red] Failed to undo: {str(e)}", style="red")
            break

def try_handle_add_command(user_input: str) -> bool:
    """Handle the /add command to include a file in the conversation."""
    prefix = "/add "
    if user_input.strip().lower().startswith(prefix):
        file_path = user_input[len(prefix):].strip()
        try:
            content = read_local_file(file_path)
            conversation_history.append({
                "role": "system",
                "content": f"Content of file '{file_path}':\n\n{content}"
            })
            console.print(f"[green]✓[/green] Added file '[cyan]{file_path}[/cyan]' to conversation.\n")
        except OSError as e:
            console.print(f"[red]✗[/red] Could not add file '[cyan]{file_path}[/cyan]': {e}\n", style="red")
        return True
    return False

def ensure_file_in_context(file_path: str) -> bool:
    """Ensure the file content is in the conversation context."""
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in msg["content"] for msg in conversation_history):
            conversation_history.append({
                "role": "system",
                "content": f"{file_marker}:\n\n{content}"
            })
        return True
    except OSError:
        console.print(f"[red]✗[/red] Could not read file '[cyan]{file_path}[/cyan]' for editing context", style="red")
        return False

def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path."""
    return str(Path(path_str).resolve())

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history = [
    {"role": "system", "content": system_PROMPT}
]

# --------------------------------------------------------------------------------
# 6. OpenAI API interaction with streaming
# --------------------------------------------------------------------------------

def guess_files_in_message(user_message: str) -> List[str]:
    """Guess which files the user might be referencing."""
    recognized_extensions = [".css", ".html", ".js", ".py", ".json", ".md"]
    potential_paths = []
    for word in user_message.split():
        if any(ext in word for ext in recognized_extensions) or "/" in word:
            path = word.strip("',\"")
            try:
                normalized_path = normalize_path(path)
                potential_paths.append(normalized_path)
            except (OSError, ValueError):
                continue
    return potential_paths

def stream_openai_response(user_message: str):
    """Stream the DeepSeek chat completion response and handle structured output."""
    potential_paths = guess_files_in_message(user_message)
    valid_files = {}

    for path in potential_paths:
        try:
            content = read_local_file(path)
            valid_files[path] = content
            file_marker = f"Content of file '{path}'"
            if not any(file_marker in msg["content"] for msg in conversation_history):
                conversation_history.append({
                    "role": "system",
                    "content": f"{file_marker}:\n\n{content}"
                })
        except OSError:
            error_msg = f"Cannot proceed: File '{path}' does not exist or is not accessible"
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            continue

    conversation_history.append({"role": "user", "content": user_message})

    try:
        stream = client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation_history,
            response_format={"type": "json_object"},
            max_completion_tokens=8000,
            stream=True
        )

        console.print("\nAssistant> ", style="bold blue", end="")
        full_content = ""

        for chunk in stream:
            if chunk.choices[0].delta.content:
                content_chunk = chunk.choices[0].delta.content
                full_content += content_chunk
                console.print(content_chunk, end="")

        console.print()

        try:
            parsed_response = json.loads(full_content)
            if "assistant_reply" not in parsed_response:
                parsed_response["assistant_reply"] = ""

            if "files_to_edit" in parsed_response and parsed_response["files_to_edit"]:
                new_files_to_edit = []
                for edit in parsed_response["files_to_edit"]:
                    try:
                        edit_abs_path = normalize_path(edit["path"])
                        if edit_abs_path in valid_files or ensure_file_in_context(edit_abs_path):
                            edit["path"] = edit_abs_path
                            new_files_to_edit.append(edit)
                    except (OSError, ValueError):
                        console.print(f"[yellow]⚠[/yellow] Skipping invalid path: '{edit['path']}'", style="yellow")
                        continue
                parsed_response["files_to_edit"] = new_files_to_edit

            response_obj = AssistantResponse(**parsed_response)
            conversation_history.append({
                "role": "assistant",
                "content": response_obj.assistant_reply
            })

            return response_obj

        except json.JSONDecodeError:
            error_msg = "Failed to parse JSON response from assistant"
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            return AssistantResponse(
                assistant_reply=error_msg,
                files_to_create=[]
            )

    except Exception as e:
        error_msg = f"DeepSeek API error: {str(e)}"
        console.print(f"\n[red]✗[/red] {error_msg}", style="red")
        return AssistantResponse(
            assistant_reply=error_msg,
            files_to_create=[]
        )

# --------------------------------------------------------------------------------
# 7. Main interactive loop
# --------------------------------------------------------------------------------

def main():
    console.print(Panel.fit(
        "[bold blue]Welcome to Deep Seek Engineer with Structured Output[/bold blue] [green](and streaming)[/green]!🐋",
        border_style="blue"
    ))
    console.print(
        "To include a file in the conversation, use '[bold magenta]/add path/to/file[/bold magenta]'.\n"
        "Type '[bold red]exit[/bold red]' or '[bold red]quit[/bold red]' to end.\n"
        "To undo changes, use '[bold magenta]/undo[/bold magenta]' or '[bold magenta]/undo N[/bold magenta]' to undo N changes.\n"
    )

    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        # Handle /add command
        if try_handle_add_command(user_input):
            continue

        # Handle /undo command
        if user_input.lower().startswith("/undo"):
            try:
                num_undos = int(user_input.split()[1]) if len(user_input.split()) > 1 else 1
                undo_last_change(num_undos)
            except ValueError:
                console.print("[red]✗[/red] Invalid number of undos. Usage: /undo [N]", style="red")
            continue

        # Get streaming response from OpenAI (DeepSeek)
        response_data = stream_openai_response(user_input)

        # Create any files if requested
        if response_data.files_to_create:
            for file_info in response_data.files_to_create:
                create_file(file_info.path, file_info.content)

        # Show and confirm diff edits if requested
        if response_data.files_to_edit:
            show_diff_table(response_data.files_to_edit)
            confirm = console.input(
                "\nDo you want to apply these changes? ([green]y[/green]/[red]n[/red]): "
            ).strip().lower()
            if confirm == 'y':
                for edit_info in response_data.files_to_edit:
                    apply_diff_edit(edit_info.path, edit_info.original_snippet, edit_info.new_snippet)
            else:
                console.print("[yellow]ℹ[/yellow] Skipped applying diff edits.", style="yellow")

    console.print("[blue]Session finished.[/blue]")

if __name__ == "__main__":
    main()
