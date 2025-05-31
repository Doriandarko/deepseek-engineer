#!/usr/bin/env python3

import os
import json
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional
from ollama import chat
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle

# Ollama settings
OLLAMA_MODEL = "deepseek-r1"
CONTEXT_LENGTH = 8192


# Initialize Rich console and prompt session
console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict(
        {
            "prompt": "#00aa00 bold",  # Green prompt
        }
    )
)


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
# 3. system prompt
# --------------------------------------------------------------------------------
system_PROMPT = dedent(
    """\
    # Role:
    You are an elite software engineer called DeepSeek Engineer with decades of experience across all programming domains.
    Your expertise spans system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning.
    You always answer using a valid JSON.
    
    #  Core capabilities:
    1. Code Analysis & Discussion
       - Analyze code with expert-level insight
       - Explain complex concepts clearly
       - Suggest optimizations and best practices
       - Debug issues with precision

    2. File Operations:
       a) Read existing files
          - Access user-provided file contents for context
          - Analyze multiple files to understand project structure
       
       b) Create new files
          - Generate complete new files with proper structure
          - Create complementary files (tests, configs, etc.)
       
       c) Edit existing files
          - Make precise changes using diff-based editing
          - Modify specific sections while preserving context
          - Suggest refactoring improvements

    # Guidelines:
    1. YOU ONLY RETURN JSON, NO OTHER TEXT OR EXPLANATION OUTSIDE THE JSON!!!
    2. For normal responses, use 'assistant_reply'
    3. When creating files, include full content in 'files_to_create'
    4. For editing files:
       - Use 'files_to_edit' for precise changes
       - Include enough context in original_snippet to locate the change
       - Ensure new_snippet maintains proper indentation
       - Prefer targeted edits over full file replacements
    5. Always explain your changes and reasoning
    6. Consider edge cases and potential impacts
    7. Follow language-specific best practices
    8. Suggest tests or validation steps when appropriate

    Remember: You're a senior engineer - be thorough, precise, and thoughtful in your solutions.
    
    # Output Format:
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
"""
)

# --------------------------------------------------------------------------------
# 4. Helper functions
# --------------------------------------------------------------------------------


def read_local_file(file_path: str) -> str:
    """Return the text content of a local file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_file(path: str, content: str):
    """Create (or overwrite) a file at 'path' with the given 'content'."""
    file_path = Path(path)

    # Security checks
    if any(part.startswith("~") for part in file_path.parts):
        raise ValueError("Home directory references not allowed")
    normalized_path = normalize_path(str(file_path))

    # Validate reasonable file size for operations
    if len(content) > 5_000_000:  # 5MB limit
        raise ValueError("File content exceeds 5MB size limit")

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(
        f"[green]✓[/green] Created/updated file at '[cyan]{file_path}[/cyan]'"
    )

    # Record the action as a system message
    conversation_history.append(
        {
            "role": "system",
            "content": f"File operation: Created/updated file at '{file_path}'",
        }
    )

    normalized_path = normalize_path(str(file_path))
    conversation_history.append(
        {
            "role": "system",
            "content": f"Content of file '{normalized_path}':\n\n{content}",
        }
    )


def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    if not files_to_edit:
        return

    table = Table(
        title="Proposed Edits",
        show_header=True,
        header_style="bold magenta",
        show_lines=True,
    )
    table.add_column("File Path", style="cyan")
    table.add_column("Original", style="red")
    table.add_column("New", style="green")

    for edit in files_to_edit:
        table.add_row(edit.path, edit.original_snippet, edit.new_snippet)

    console.print(table)


def apply_diff_edit(path: str, original_snippet: str, new_snippet: str):
    """Reads the file at 'path', replaces the first occurrence of 'original_snippet' with 'new_snippet', then overwrites."""
    try:
        content = read_local_file(path)

        # Verify we're replacing the exact intended occurrence
        occurrences = content.count(original_snippet)
        if occurrences == 0:
            raise ValueError("Original snippet not found")
        if occurrences > 1:
            console.print(
                f"[yellow]Multiple matches ({occurrences}) found - requiring line numbers for safety",
                style="yellow",
            )
            console.print("Use format:\n--- original.py (lines X-Y)\n+++ modified.py\n")
            raise ValueError(f"Ambiguous edit: {occurrences} matches")

        updated_content = content.replace(original_snippet, new_snippet, 1)
        create_file(path, updated_content)
        console.print(f"[green]✓[/green] Applied diff edit to '[cyan]{path}[/cyan]'")
        # Record the edit as a system message
        conversation_history.append(
            {
                "role": "system",
                "content": f"File operation: Applied diff edit to '{path}'",
            }
        )
    except FileNotFoundError:
        console.print(
            f"[red]✗[/red] File not found for diff editing: '[cyan]{path}[/cyan]'",
            style="red",
        )
    except ValueError as e:
        console.print(
            f"[yellow]⚠[/yellow] {str(e)} in '[cyan]{path}[/cyan]'. No changes made.",
            style="yellow",
        )
        console.print("\nExpected snippet:", style="yellow")
        console.print(Panel(original_snippet, title="Expected", border_style="yellow"))
        console.print("\nActual file content:", style="yellow")
        console.print(Panel(content, title="Actual", border_style="yellow"))


def try_handle_add_command(user_input: str) -> bool:
    prefix = "/add "
    if user_input.strip().lower().startswith(prefix):
        path_to_add = user_input[len(prefix) :].strip()
        try:
            normalized_path = normalize_path(path_to_add)
            if os.path.isdir(normalized_path):
                # Handle entire directory
                add_directory_to_conversation(normalized_path)
            else:
                # Handle a single file as before
                content = read_local_file(normalized_path)
                conversation_history.append(
                    {
                        "role": "system",
                        "content": f"Content of file '{normalized_path}':\n\n{content}",
                    }
                )
                console.print(
                    f"[green]✓[/green] Added file '[cyan]{normalized_path}[/cyan]' to conversation.\n"
                )
        except OSError as e:
            console.print(
                f"[red]✗[/red] Could not add path '[cyan]{path_to_add}[/cyan]': {e}\n",
                style="red",
            )
        return True
    return False


def add_directory_to_conversation(directory_path: str):
    with console.status("[bold green]Scanning directory...") as status:
        excluded_files = {
            # Python specific
            ".DS_Store",
            "Thumbs.db",
            ".gitignore",
            ".python-version",
            "uv.lock",
            ".uv",
            "uvenv",
            ".uvenv",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".coverage",
            ".mypy_cache",
            # Node.js / Web specific
            "node_modules",
            "package-lock.json",
            "yarn.lock",
            "pnpm-lock.yaml",
            ".next",
            ".nuxt",
            "dist",
            "build",
            ".cache",
            ".parcel-cache",
            ".turbo",
            ".vercel",
            ".output",
            ".contentlayer",
            # Build outputs
            "out",
            "coverage",
            ".nyc_output",
            "storybook-static",
            # Environment and config
            ".env",
            ".env.local",
            ".env.development",
            ".env.production",
            # Misc
            ".git",
            ".svn",
            ".hg",
            "CVS",
        }
        excluded_extensions = {
            # Binary and media files
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".svg",
            ".webp",
            ".avif",
            ".mp4",
            ".webm",
            ".mov",
            ".mp3",
            ".wav",
            ".ogg",
            ".zip",
            ".tar",
            ".gz",
            ".7z",
            ".rar",
            ".exe",
            ".dll",
            ".so",
            ".dylib",
            ".bin",
            # Documents
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            # Python specific
            ".pyc",
            ".pyo",
            ".pyd",
            ".egg",
            ".whl",
            # UV specific
            ".uv",
            ".uvenv",
            # Database and logs
            ".db",
            ".sqlite",
            ".sqlite3",
            ".log",
            # IDE specific
            ".idea",
            ".vscode",
            # Web specific
            ".map",
            ".chunk.js",
            ".chunk.css",
            ".min.js",
            ".min.css",
            ".bundle.js",
            ".bundle.css",
            # Cache and temp files
            ".cache",
            ".tmp",
            ".temp",
            # Font files
            ".ttf",
            ".otf",
            ".woff",
            ".woff2",
            ".eot",
        }
        skipped_files = []
        added_files = []
        total_files_processed = 0
        max_files = 1000  # Reasonable limit for files to process
        max_file_size = 5_000_000  # 5MB limit

        for root, dirs, files in os.walk(directory_path):
            if total_files_processed >= max_files:
                console.print(
                    f"[yellow]⚠[/yellow] Reached maximum file limit ({max_files})"
                )
                break

            status.update(f"[bold green]Scanning {root}...")
            # Skip hidden directories and excluded directories
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and d not in excluded_files
            ]

            for file in files:
                if total_files_processed >= max_files:
                    break

                if file.startswith(".") or file in excluded_files:
                    skipped_files.append(os.path.join(root, file))
                    continue

                _, ext = os.path.splitext(file)
                if ext.lower() in excluded_extensions:
                    skipped_files.append(os.path.join(root, file))
                    continue

                full_path = os.path.join(root, file)

                try:
                    # Check file size before processing
                    if os.path.getsize(full_path) > max_file_size:
                        skipped_files.append(f"{full_path} (exceeds size limit)")
                        continue

                    # Check if it's binary
                    if is_binary_file(full_path):
                        skipped_files.append(full_path)
                        continue

                    normalized_path = normalize_path(full_path)
                    content = read_local_file(normalized_path)
                    conversation_history.append(
                        {
                            "role": "system",
                            "content": f"Content of file '{normalized_path}':\n\n{content}",
                        }
                    )
                    added_files.append(normalized_path)
                    total_files_processed += 1

                except OSError:
                    skipped_files.append(full_path)

        console.print(
            f"[green]✓[/green] Added folder '[cyan]{directory_path}[/cyan]' to conversation."
        )
        if added_files:
            console.print(
                f"\n[bold]Added files:[/bold] ({len(added_files)} of {total_files_processed})"
            )
            for f in added_files:
                console.print(f"[cyan]{f}[/cyan]")
        if skipped_files:
            console.print(f"\n[yellow]Skipped files:[/yellow] ({len(skipped_files)})")
            for f in skipped_files:
                console.print(f"[yellow]{f}[/yellow]")
        console.print()


def is_binary_file(file_path: str, peek_size: int = 1024) -> bool:
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(peek_size)
        # If there is a null byte in the sample, treat it as binary
        if b"\0" in chunk:
            return True
        return False
    except Exception:
        # If we fail to read, just treat it as binary to be safe
        return True


def ensure_file_in_context(file_path: str) -> bool:
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        file_marker = f"Content of file '{normalized_path}'"
        if not any(file_marker in msg["content"] for msg in conversation_history):
            conversation_history.append(
                {"role": "system", "content": f"{file_marker}:\n\n{content}"}
            )
        return True
    except OSError:
        console.print(
            f"[red]✗[/red] Could not read file '[cyan]{file_path}[/cyan]' for editing context",
            style="red",
        )
        return False


def normalize_path(path_str: str) -> str:
    """Return a canonical, absolute version of the path with security checks."""
    path = Path(path_str).resolve()

    # Prevent directory traversal attacks
    if ".." in path.parts:
        raise ValueError(
            f"Invalid path: {path_str} contains parent directory references"
        )

    return str(path)


# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history = [{"role": "system", "content": system_PROMPT}]

# --------------------------------------------------------------------------------
# 6. OpenAI API interaction with streaming
# --------------------------------------------------------------------------------


def guess_files_in_message(user_message: str) -> List[str]:
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
    # First, clean up the conversation history while preserving system messages with file content
    system_msgs = [conversation_history[0]]  # Keep initial system prompt
    file_context = []
    user_assistant_pairs = []

    for msg in conversation_history[1:]:
        if msg["role"] == "system" and "Content of file '" in msg["content"]:
            file_context.append(msg)
        elif msg["role"] in ["user", "assistant"]:
            user_assistant_pairs.append(msg)

    # Only keep complete user-assistant pairs
    if len(user_assistant_pairs) % 2 != 0:
        user_assistant_pairs = user_assistant_pairs[:-1]

    # Rebuild clean history with files preserved
    cleaned_history = system_msgs + file_context
    cleaned_history.extend(user_assistant_pairs)
    cleaned_history.append({"role": "user", "content": user_message})

    # Replace conversation_history with cleaned version
    conversation_history.clear()
    conversation_history.extend(cleaned_history)

    potential_paths = guess_files_in_message(user_message)
    valid_files = {}

    for path in potential_paths:
        try:
            content = read_local_file(path)
            valid_files[path] = content
            file_marker = f"Content of file '{path}'"
            if not any(file_marker in msg["content"] for msg in conversation_history):
                conversation_history.append(
                    {"role": "system", "content": f"{file_marker}:\n\n{content}"}
                )
        except OSError:
            error_msg = (
                f"Cannot proceed: File '{path}' does not exist or is not accessible"
            )
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            continue

    try:
        stream = chat(
            model=OLLAMA_MODEL,
            messages=conversation_history,
            format="",  #'json'
            stream=True,
            options={"temperature": 0, "num_ctx": CONTEXT_LENGTH},
        )

        console.print("\nThinking...", style="bold yellow")
        reasoning_started = False
        reasoning_content = ""
        final_content = ""

        for chunk in stream:
            if "<think>" in chunk["message"]["content"]:
                reasoning_started = True
                console.print("\nReasoning:", style="bold yellow")
                continue
            if reasoning_started:
                console.print(chunk["message"]["content"], end="")
                reasoning_content += chunk["message"]["content"]
            if "</think>" in chunk["message"]["content"]:
                reasoning_content = reasoning_content.replace("</think>", "")
                reasoning_started = False
                console.print("\n")  # Add spacing after reasoning
                console.print("\nAssistant> ", style="bold blue", end="")
                continue
            if not reasoning_started:
                console.print(chunk["message"]["content"], end="")
                final_content += chunk["message"]["content"]
        console.print()  # New line after streaming

        try:
            for attempt in range(1, 4):
                try:
                    if attempt == 1:
                        parsed_response = json.loads(final_content)
                    elif attempt == 2:
                        final_content = extract_json_string(final_content)
                        parsed_response = json.loads(final_content)
                    elif attempt == 3:
                        console.print("Parsing result:", style="bold yellow")
                        final_content = recover_json_AI(final_content)
                        parsed_response = json.loads(final_content)
                    break
                except:
                    print(f"Attempt {attempt} failed")
                    if attempt == 3:
                        raise json.JSONDecodeError

            if "assistant_reply" not in parsed_response:
                parsed_response["assistant_reply"] = ""

            if "files_to_edit" in parsed_response and parsed_response["files_to_edit"]:
                new_files_to_edit = []
                for edit in parsed_response["files_to_edit"]:
                    try:
                        edit_abs_path = normalize_path(edit["path"])
                        if edit_abs_path in valid_files or ensure_file_in_context(
                            edit_abs_path
                        ):
                            edit["path"] = edit_abs_path
                            new_files_to_edit.append(edit)
                    except (OSError, ValueError):
                        console.print(
                            f"[yellow]⚠[/yellow] Skipping invalid path: '{edit['path']}'",
                            style="yellow",
                        )
                        continue
                parsed_response["files_to_edit"] = new_files_to_edit

            response_obj = AssistantResponse(**parsed_response)

            # Store the complete JSON response in conversation history
            conversation_history.append(
                {
                    "role": "assistant",
                    "content": final_content,  # Store the full JSON response string
                }
            )

            return response_obj

        except json.JSONDecodeError:
            error_msg = "Failed to parse JSON response from assistant"
            console.print(f"[red]✗[/red] {error_msg}", style="red")
            return AssistantResponse(assistant_reply=error_msg, files_to_create=[])

    except Exception as e:
        error_msg = f"DeepSeek API error: {str(e)}"
        console.print(f"\n[red]✗[/red] {error_msg}", style="red")
        return AssistantResponse(assistant_reply=error_msg, files_to_create=[])


def extract_json_string(text):
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and start < end:
        return text[start : end + 1]
    else:
        return text


def recover_json_AI(final_content) -> str:
    system_prompt = """
    # Role:
        Read the following **instructions** and convert them to a valid JSON following the **Json_Schema**.
        Only answer with a valid Json.
        
    # Json_Schema:
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
    """
    user_prompt = dedent(
        f"""
    <instructions>
    {final_content}
    </instructions>
    """
    )

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    stream = chat(
        model=OLLAMA_MODEL,
        messages=conversation_history,
        format="",  #'json'
        stream=True,
        options={"temperature": 0, "num_ctx": CONTEXT_LENGTH},
    )

    response = ""
    for chunk in stream:
        console.print(chunk["message"]["content"], end="")
        response += chunk["message"]["content"]
    console.print()  # New line after streaming
    if "</think>" in response:
        response = (
            response[response.rfind("</think>") :].replace("</think>", "").lstrip()
        )
        response = extract_json_string(response)
    return response


def trim_conversation_history():
    """Trim conversation history to prevent token limit issues"""
    max_pairs = 10  # Adjust based on your needs
    system_msgs = [msg for msg in conversation_history if msg["role"] == "system"]
    other_msgs = [msg for msg in conversation_history if msg["role"] != "system"]

    # Keep only the last max_pairs of user-assistant interactions
    if len(other_msgs) > max_pairs * 2:
        other_msgs = other_msgs[-max_pairs * 2 :]

    conversation_history.clear()
    conversation_history.extend(system_msgs + other_msgs)


# --------------------------------------------------------------------------------
# 7. Main interactive loop
# --------------------------------------------------------------------------------


def main():
    console.print(
        Panel.fit(
            "[bold blue]Welcome to Deep Seek Engineer with Structured Output[/bold blue] [green](and CoT reasoning)[/green]!🐋",
            border_style="blue",
        )
    )
    console.print(
        "Use '[bold magenta]/add[/bold magenta]' to include files in the conversation:\n"
        "  • '[bold magenta]/add path/to/file[/bold magenta]' for a single file\n"
        "  • '[bold magenta]/add path/to/folder[/bold magenta]' for all files in a folder\n"
        "  • You can add multiple files one by one using /add for each file\n"
        "Type '[bold red]exit[/bold red]' or '[bold red]quit[/bold red]' to end.\n"
    )

    while True:
        try:
            user_input = prompt_session.prompt("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[yellow]Exiting.[/yellow]")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("[yellow]Goodbye![/yellow]")
            break

        if try_handle_add_command(user_input):
            continue

        response_data = stream_openai_response(user_input)

        if response_data.files_to_create:
            for file_info in response_data.files_to_create:
                create_file(file_info.path, file_info.content)

        if response_data.files_to_edit:
            show_diff_table(response_data.files_to_edit)
            confirm = (
                prompt_session.prompt("Do you want to apply these changes? (y/n): ")
                .strip()
                .lower()
            )
            if confirm == "y":
                for edit_info in response_data.files_to_edit:
                    apply_diff_edit(
                        edit_info.path,
                        edit_info.original_snippet,
                        edit_info.new_snippet,
                    )
            else:
                console.print(
                    "[yellow]ℹ[/yellow] Skipped applying diff edits.", style="yellow"
                )

    console.print("[blue]Session finished.[/blue]")


if __name__ == "__main__":
    main()
