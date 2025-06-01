#!/usr/bin/env python3

import os
import sys
import json
import re
import time
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta # ChoiceDeltaToolCall is implicitly handled by iterating delta.tool_calls
from openai.types.completion_usage import CompletionUsage 
from pydantic import BaseModel
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.style import Style
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle

# Initialize Rich console and prompt session
console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#0066ff bold',
        'completion-menu.completion': 'bg:#1e3a8a fg:#ffffff',
        'completion-menu.completion.current': 'bg:#3b82f6 fg:#ffffff bold',
    })
)

# Global base directory for operations (default: current working directory)
base_dir = Path.cwd()

# --------------------------------------------------------------------------------
# 0. Configuration Constants
# --------------------------------------------------------------------------------
MAX_FILES_IN_ADD_DIR = 1000
MAX_FILE_SIZE_IN_ADD_DIR = 5_000_000
MAX_FILE_CONTENT_SIZE_CREATE = 5_000_000

ADD_COMMAND_PREFIX = "/add "
COMMIT_COMMAND_PREFIX = "/commit "
GIT_BRANCH_COMMAND_PREFIX = "/git branch "

git_context = {
    'enabled': False,
    'skip_staging': False,
    'branch': None
}

# --------------------------------------------------------------------------------
# 1. Configure OpenAI client and load environment variables
# --------------------------------------------------------------------------------
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# --------------------------------------------------------------------------------
# 2. Define Pydantic models
# --------------------------------------------------------------------------------
class FileToCreate(BaseModel):
    path: str
    content: str

class FileToEdit(BaseModel):
    path: str
    original_snippet: str
    new_snippet: str

# --------------------------------------------------------------------------------
# 2.1. Define Function Calling Tools
# --------------------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the content of a single file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "The path to the file to read"}},
                "required": ["file_path"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Read the content of multiple files",
            "parameters": {
                "type": "object",
                "properties": {"file_paths": {"type": "array", "items": {"type": "string"}, "description": "Array of file paths to read"}},
                "required": ["file_paths"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create or overwrite a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path for the file"},
                    "content": {"type": "string", "description": "Content for the file"}
                },
                "required": ["file_path", "content"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_multiple_files",
            "description": "Create multiple files",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["path", "content"]
                        },
                        "description": "Array of files to create (path, content)",
                    }
                },
                "required": ["files"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file by replacing a snippet",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the file"},
                    "original_snippet": {"type": "string", "description": "Snippet to replace"},
                    "new_snippet": {"type": "string", "description": "Replacement snippet"}
                },
                "required": ["file_path", "original_snippet", "new_snippet"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_init",
            "description": "Initialize a new Git repository.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Commit staged changes with a message.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string", "description": "Commit message"}},
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_create_branch",
            "description": "Create and switch to a new Git branch.",
            "parameters": {
                "type": "object",
                "properties": {"branch_name": {"type": "string", "description": "Name of the new branch"}},
                "required": ["branch_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show current Git status.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_add",
            "description": "Stage files for commit.",
            "parameters": {
                "type": "object",
                "properties": {"file_paths": {"type": "array", "items": {"type": "string"}, "description": "Paths of files to stage"}},
                "required": ["file_paths"]
            }
        }
    }
]

# --------------------------------------------------------------------------------
# 3. System prompt
# --------------------------------------------------------------------------------
system_PROMPT = dedent("""\
    You are an elite software engineer called DeepSeek Engineer with decades of experience across all programming domains.
    Your expertise spans system design, algorithms, testing, and best practices.
    You provide thoughtful, well-structured solutions while explaining your reasoning.

    Core capabilities:
    1. Code Analysis & Discussion
       - Analyze code with expert-level insight
       - Explain complex concepts clearly
       - Suggest optimizations and best practices
       - Debug issues with precision

    2. File Operations (via function calls):
       - read_file: Read a single file's content
       - read_multiple_files: Read multiple files at once
       - create_file: Create or overwrite a single file
       - create_multiple_files: Create multiple files at once
       - edit_file: Make precise edits to existing files using snippet replacement

    3. Git Operations (via function calls):
       - git_init: Initialize a new Git repository in the current directory.
       - git_add: Stage specified file(s) for the next commit. Use this before git_commit.
       - git_commit: Commit staged changes with a message. Ensure files are staged first using git_add.
       - git_create_branch: Create and switch to a new Git branch.
       - git_status: Show the current Git status, useful for seeing what is staged or unstaged.

    Guidelines:
    1. Provide natural, conversational responses explaining your reasoning
    2. Use function calls when you need to read or modify files, or interact with Git.
    3. For file operations:
       - Always read files first before editing them to understand the context
       - Use precise snippet matching for edits
       - Explain what changes you're making and why
       - Consider the impact of changes on the overall codebase
    4. For Git operations:
       - Use `git_add` to stage files before using `git_commit`.
       - Provide clear commit messages.
       - Check `git_status` if unsure about the state of the repository.
    5. Follow language-specific best practices
    6. Suggest tests or validation steps when appropriate
    7. Be thorough in your analysis and recommendations

    IMPORTANT: In your thinking process, if you realize that something requires a tool call, cut your thinking short and proceed directly to the tool call. Don't overthink - act efficiently when file or Git operations are needed.

    Remember: You're a senior engineer - be thoughtful, precise, and explain your reasoning clearly.
""")

# --------------------------------------------------------------------------------
# 4. NEW: Smart conversation history management
# --------------------------------------------------------------------------------

def smart_truncate_history(conversation_history, max_messages=50):
    """
    Truncate conversation history while preserving tool call sequences and important context.
    """
    if len(conversation_history) <= max_messages:
        return conversation_history
    
    # Always keep system prompt at index 0 and any critical system messages
    system_messages = []
    other_messages = []
    
    for msg in conversation_history:
        if msg["role"] == "system":
            system_messages.append(msg)
        else:
            other_messages.append(msg)
    
    # Keep the main system prompt and recent file context messages
    if len(system_messages) > 1:
        # Keep first system message (main prompt) and last few file contexts
        important_system = [system_messages[0]]
        file_contexts = [msg for msg in system_messages[1:] if "User added file" in msg["content"]]
        important_system.extend(file_contexts[-3:])  # Keep last 3 file contexts
        system_messages = important_system
    
    # Work backwards to find a good truncation point for conversation flow
    keep_messages = []
    i = len(other_messages) - 1
    messages_to_keep = max_messages - len(system_messages)
    
    while i >= 0 and len(keep_messages) < messages_to_keep:
        current_msg = other_messages[i]
        
        # If this is a tool result, we need to keep the corresponding assistant message
        if current_msg["role"] == "tool":
            # Collect all tool results for this sequence
            tool_sequence = []
            while i >= 0 and other_messages[i]["role"] == "tool":
                tool_sequence.insert(0, other_messages[i])
                i -= 1
            
            # Add the tool results
            keep_messages = tool_sequence + keep_messages
            
            # Find and add the corresponding assistant message with tool_calls
            if i >= 0 and other_messages[i]["role"] == "assistant" and other_messages[i].get("tool_calls"):
                keep_messages.insert(0, other_messages[i])
                i -= 1
        else:
            # Regular message (user or assistant)
            keep_messages.insert(0, current_msg)
            i -= 1
    
    # Combine system messages with kept conversation messages
    result = system_messages + keep_messages
    
    # Final safety check - if still too long, trim more aggressively but keep recent complete sequences
    if len(result) > max_messages:
        excess = len(result) - max_messages
        # Remove from the middle, keeping system messages and recent conversation
        system_count = len(system_messages)
        result = system_messages + keep_messages[excess:]
    
    return result

def validate_tool_calls(accumulated_tool_calls):
    """
    Validate accumulated tool calls and provide debugging info.
    """
    if not accumulated_tool_calls:
        return []
    
    valid_calls = []
    for i, tool_call in enumerate(accumulated_tool_calls):
        # Check for required fields
        if not tool_call.get("id"):
            console.print(f"[yellow]⚠ Tool call {i} missing ID, skipping[/yellow]")
            continue
        
        func_name = tool_call.get("function", {}).get("name")
        if not func_name:
            console.print(f"[yellow]⚠ Tool call {i} missing function name, skipping[/yellow]")
            continue
        
        func_args = tool_call.get("function", {}).get("arguments", "")
        
        # Validate JSON arguments
        try:
            if func_args:
                json.loads(func_args)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗ Tool call {i} has invalid JSON arguments: {e}[/red]")
            console.print(f"[red]  Arguments: {func_args}[/red]")
            continue
        
        valid_calls.append(tool_call)
    
    if len(valid_calls) != len(accumulated_tool_calls):
        console.print(f"[yellow]⚠ Kept {len(valid_calls)}/{len(accumulated_tool_calls)} tool calls[/yellow]")
    
    return valid_calls

def add_file_context_smartly(conversation_history, file_path, content, max_context_files=5):
    """
    Add file context while managing system message bloat and avoiding duplicates.
    """
    marker = f"User added file '{file_path}'"
    
    # Remove any existing context for this exact file to avoid duplicates
    conversation_history[:] = [
        msg for msg in conversation_history 
        if not (msg["role"] == "system" and marker in msg["content"])
    ]
    
    # Count existing file context messages
    file_context_count = sum(
        1 for msg in conversation_history 
        if msg["role"] == "system" and "User added file" in msg["content"]
    )
    
    # If too many file contexts, remove oldest ones (but keep the main system prompt)
    if file_context_count >= max_context_files:
        removed_count = 0
        new_history = []
        
        for msg in conversation_history:
            if (msg["role"] == "system" and 
                "User added file" in msg["content"] and 
                removed_count < (file_context_count - max_context_files + 1)):
                removed_count += 1
                continue  # Skip this old file context
            new_history.append(msg)
        
        conversation_history[:] = new_history
    
    # Add new file context
    conversation_history.append({
        "role": "system", 
        "content": f"{marker}. Content:\n\n{content}"
    })

# --------------------------------------------------------------------------------
# 4. Helper functions (updated)
# --------------------------------------------------------------------------------

def read_local_file(file_path: str) -> str:
    full_path = (base_dir / file_path).resolve()
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

def create_file(path: str, content: str):
    file_path = Path(path)
    if any(part.startswith('~') for part in file_path.parts):
        raise ValueError("Home directory references not allowed")
    normalized_path_str = normalize_path(str(file_path)) 
    
    if len(content) > MAX_FILE_CONTENT_SIZE_CREATE:
        raise ValueError(f"File content exceeds {MAX_FILE_CONTENT_SIZE_CREATE // (1024*1024)}MB size limit")
    
    Path(normalized_path_str).parent.mkdir(parents=True, exist_ok=True)
    with open(normalized_path_str, "w", encoding="utf-8") as f:
        f.write(content)
    console.print(f"[bold blue]✓[/bold blue] Created/updated file at '[bright_cyan]{normalized_path_str}[/bright_cyan]'")
    
    if git_context['enabled'] and not git_context['skip_staging']:
        stage_file(normalized_path_str)

def show_diff_table(files_to_edit: List[FileToEdit]) -> None:
    if not files_to_edit: return
    table = Table(title="📝 Proposed Edits", show_header=True, header_style="bold bright_blue", show_lines=True, border_style="blue")
    table.add_column("File Path", style="bright_cyan", no_wrap=True)
    table.add_column("Original", style="red dim")
    table.add_column("New", style="bright_green")
    for edit in files_to_edit: table.add_row(edit.path, edit.original_snippet, edit.new_snippet)
    console.print(table)

def apply_diff_edit(path: str, original_snippet: str, new_snippet: str):
    content = ""
    try:
        normalized_path_str = normalize_path(path)
        content = read_local_file(normalized_path_str)
        occurrences = content.count(original_snippet)
        if occurrences == 0:
            raise ValueError("Original snippet not found")
        if occurrences > 1:
            console.print(f"[bold yellow]⚠ Multiple matches ({occurrences}) found. Aborting.[/bold yellow]")
            raise ValueError(f"Ambiguous edit: {occurrences} matches.")
        
        updated_content = content.replace(original_snippet, new_snippet, 1)
        create_file(normalized_path_str, updated_content)
        console.print(f"[bold blue]✓[/bold blue] Applied diff edit to '[bright_cyan]{normalized_path_str}[/bright_cyan]'")

    except FileNotFoundError:
        console.print(f"[bold red]✗[/bold red] File not found for diff: '[bright_cyan]{path}[/bright_cyan]'")
    except ValueError as e:
        console.print(f"[bold yellow]⚠[/bold yellow] {str(e)} in '[bright_cyan]{path}[/bright_cyan]'. No changes.")
        if "Original snippet not found" in str(e) or "Ambiguous edit" in str(e):
            console.print("\n[bold blue]Expected snippet:[/bold blue]")
            console.print(Panel(original_snippet, title="Expected", border_style="blue"))
            if content:
                console.print("\n[bold blue]Actual content (or relevant part):[/bold blue]")
                start_idx = max(0, content.find(original_snippet[:20]) - 100)
                end_idx = min(len(content), start_idx + len(original_snippet) + 200)
                display_snip = ("..." if start_idx > 0 else "") + content[start_idx:end_idx] + ("..." if end_idx < len(content) else "")
                console.print(Panel(display_snip or content, title="Actual", border_style="yellow"))

def try_handle_add_command(user_input: str) -> bool:
    if user_input.strip().lower().startswith(ADD_COMMAND_PREFIX):
        path_to_add = user_input[len(ADD_COMMAND_PREFIX):].strip()
        try:
            normalized_path = normalize_path(path_to_add)
            if not Path(normalized_path).exists():
                console.print(f"[bold red]✗[/bold red] Path does not exist: '[bright_cyan]{path_to_add}[/bright_cyan]'")
                return True
            if Path(normalized_path).is_dir():
                add_directory_to_conversation(normalized_path)
            else:
                content = read_local_file(normalized_path)
                add_file_context_smartly(conversation_history, normalized_path, content)
                console.print(f"[bold blue]✓[/bold blue] Added file '[bright_cyan]{normalized_path}[/bright_cyan]' to conversation.\n")
        except (OSError, ValueError) as e:
            console.print(f"[bold red]✗[/bold red] Could not add path '[bright_cyan]{path_to_add}[/bright_cyan]': {e}\n")
        return True
    return False

def try_handle_commit_command(user_input: str) -> bool:
    if user_input.strip().lower().startswith(COMMIT_COMMAND_PREFIX.strip()):
        if not git_context['enabled']:
            console.print("[yellow]Git not enabled. `/git init` first.[/yellow]")
            return True
        message = user_input[len(COMMIT_COMMAND_PREFIX.strip()):].strip()
        if user_input.strip().lower() == COMMIT_COMMAND_PREFIX.strip() and not message:
            message = prompt_session.prompt("🔵 Enter commit message: ").strip()
            if not message:
                console.print("[yellow]Commit aborted. Message empty.[/yellow]")
                return True
        elif not message:
            console.print("[yellow]Provide commit message: /commit <message>[/yellow]")
            return True
        user_commit_changes(message)
        return True
    return False

def try_handle_git_command(user_input: str) -> bool:
    cmd = user_input.strip().lower()
    if cmd == "/git init": return initialize_git_repo_cmd()
    elif cmd.startswith(GIT_BRANCH_COMMAND_PREFIX.strip()):
        branch_name = user_input[len(GIT_BRANCH_COMMAND_PREFIX.strip()):].strip()
        if not branch_name and cmd == GIT_BRANCH_COMMAND_PREFIX.strip():
             console.print("[yellow]Specify branch name: /git branch <name>[/yellow]")
             return True
        return create_git_branch_cmd(branch_name)
    elif cmd == "/git status": return show_git_status_cmd()
    return False

def try_handle_git_info_command(user_input: str) -> bool:
    if user_input.strip().lower() == "/git-info":
        console.print("I can use Git commands to interact with a Git repository. Here's what I can do for you:\n\n"
                      "1. **Initialize a Git repository**: Use `git_init` to create a new Git repository in the current directory.\n"
                      "2. **Stage files for commit**: Use `git_add` to stage specific files for the next commit.\n"
                      "3. **Commit changes**: Use `git_commit` to commit staged changes with a message.\n"
                      "4. **Create and switch to a new branch**: Use `git_create_branch` to create a new branch and switch to it.\n"
                      "5. **Check Git status**: Use `git_status` to see the current state of the repository (staged, unstaged, or untracked files).\n\n"
                      "Let me know what you'd like to do, and I can perform the necessary Git operations for you. For example:\n"
                      "- Do you want to initialize a new repository?\n"
                      "- Stage and commit changes?\n"
                      "- Create a new branch? \n\n"
                      "Just provide the details, and I'll handle the rest!")
        return True
    return False

def try_handle_clear_command(user_input: str) -> bool:
    if user_input.strip().lower() == "/clear":
        console.clear()
        return True
    return False

def try_handle_folder_command(user_input: str) -> bool:
    global base_dir
    if user_input.strip().lower().startswith("/folder"):
        folder_path = user_input[len("/folder"):].strip()
        if not folder_path:
            console.print(f"[yellow]Current base directory: '{base_dir}'[/yellow]")
            console.print("[yellow]Usage: /folder <path> or /folder reset[/yellow]")
            return True
        if folder_path.lower() == "reset":
            old_base = base_dir
            base_dir = Path.cwd()
            console.print(f"[green]✓ Base directory reset from '{old_base}' to: '{base_dir}'[/green]")
            return True
        try:
            new_base = Path(folder_path).resolve()
            if not new_base.exists() or not new_base.is_dir():
                console.print(f"[red]✗ Path does not exist or is not a directory: '{folder_path}'[/red]")
                return True
            # Check write permissions
            test_file = new_base / ".eng-git-test"
            try:
                test_file.touch()
                test_file.unlink()
            except PermissionError:
                console.print(f"[red]✗ No write permissions in directory: '{new_base}'[/red]")
                return True
            old_base = base_dir
            base_dir = new_base
            console.print(f"[green]✓ Base directory changed from '{old_base}' to: '{base_dir}'[/green]")
            console.print(f"[green]  All relative paths will now be resolved against this directory.[/green]")
            # Automatically trigger /add for the new directory
            add_directory_to_conversation(str(new_base))
            return True
        except Exception as e:
            console.print(f"[red]✗ Error setting base directory: {e}[/red]")
            return True
    return False

def try_handle_exit_command(user_input: str) -> bool:
    if user_input.strip().lower() in ("/exit", "/quit"):
        console.print("[bold blue]👋 Goodbye![/bold blue]")
        sys.exit(0)
    return False

def try_handle_help_command(user_input: str) -> bool:
    if user_input.strip().lower() == "/help":
        help_table = Table(title="📝 Available Commands", show_header=True, header_style="bold bright_blue")
        help_table.add_column("Command", style="bright_cyan"); help_table.add_column("Description", style="white")
        
        # General commands
        help_table.add_row("/help", "Show this help")
        help_table.add_row("/clear", "Clear screen")
        help_table.add_row("/exit, /quit", "Exit application")
        
        # Directory & file management
        help_table.add_row("/folder", "Show current base directory")
        help_table.add_row("/folder <path>", "Set base directory for file operations")
        help_table.add_row("/folder reset", "Reset base directory to current working directory")
        help_table.add_row(f"{ADD_COMMAND_PREFIX.strip()} <path>", "Add file/dir to conversation context")
        
        # Git workflow commands
        help_table.add_row("/git init", "Initialize Git repository")
        help_table.add_row("/git status", "Show Git status")
        help_table.add_row(f"{GIT_BRANCH_COMMAND_PREFIX.strip()} <name>", "Create & switch to new branch")
        help_table.add_row(f"{COMMIT_COMMAND_PREFIX.strip()} [msg]", "Stage all files & commit (prompts if no message)")
        help_table.add_row("/git-info", "Show detailed Git capabilities")
        
        console.print(help_table)
        return True
    return False

def add_directory_to_conversation(directory_path: str):
    with console.status("[bold bright_blue]🔍 Scanning directory...[/bold bright_blue]") as status:
        excluded_files = {".DS_Store", "Thumbs.db", ".gitignore", ".python-version", "uv.lock", ".uv", "uvenv", ".uvenv", ".venv", "venv", "__pycache__", ".pytest_cache", ".coverage", ".mypy_cache", "node_modules", "package-lock.json", "yarn.lock", "pnpm-lock.yaml", ".next", ".nuxt", "dist", "build", ".cache", ".parcel-cache", ".turbo", ".vercel", ".output", ".contentlayer", "out", "coverage", ".nyc_output", "storybook-static", ".env", ".env.local", ".env.development", ".env.production", ".git", ".svn", ".hg", "CVS"}
        excluded_extensions = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".avif", ".mp4", ".webm", ".mov", ".mp3", ".wav", ".ogg", ".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dll", ".so", ".dylib", ".bin", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pyc", ".pyo", ".pyd", ".egg", ".whl", ".uv", ".uvenv", ".db", ".sqlite", ".sqlite3", ".log", ".idea", ".vscode", ".map", ".chunk.js", ".chunk.css", ".min.js", ".min.css", ".bundle.js", ".bundle.css", ".cache", ".tmp", ".temp", ".ttf", ".otf", ".woff", ".woff2", ".eot"}
        skipped, added, total_processed = [], [], 0
        for root, dirs, files in os.walk(directory_path):
            if total_processed >= MAX_FILES_IN_ADD_DIR: console.print(f"[yellow]⚠ Max files ({MAX_FILES_IN_ADD_DIR}) reached for dir scan."); break
            status.update(f"[bold bright_blue]🔍 Scanning {root}...[/bold bright_blue]")
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_files]
            for file in files:
                if total_processed >= MAX_FILES_IN_ADD_DIR: break
                if file.startswith('.') or file in excluded_files or os.path.splitext(file)[1].lower() in excluded_extensions: skipped.append(os.path.join(root, file)); continue
                full_path = os.path.join(root, file)
                try:
                    if os.path.getsize(full_path) > MAX_FILE_SIZE_IN_ADD_DIR: skipped.append(f"{full_path} (size limit)"); continue
                    if is_binary_file(full_path): skipped.append(f"{full_path} (binary)"); continue
                    norm_path = normalize_path(full_path)
                    content = read_local_file(norm_path)
                    add_file_context_smartly(conversation_history, norm_path, content)
                    added.append(norm_path); total_processed += 1
                except (OSError, ValueError) as e: skipped.append(f"{full_path} (error: {e})")
        console.print(f"[bold blue]✓[/bold blue] Added folder '[bright_cyan]{directory_path}[/bright_cyan]'.")
        if added: console.print(f"\n[bold bright_blue]📁 Added:[/bold bright_blue] ({len(added)} of {total_processed} valid) {[Path(f).name for f in added[:5]]}{'...' if len(added) > 5 else ''}")
        if skipped: console.print(f"\n[yellow]⏭ Skipped:[/yellow] ({len(skipped)}) {[Path(f).name for f in skipped[:3]]}{'...' if len(skipped) > 3 else ''}")
        console.print()

def is_binary_file(file_path: str, peek_size: int = 1024) -> bool:
    try:
        with open(file_path, 'rb') as f: chunk = f.read(peek_size)
        return b'\0' in chunk
    except Exception: return True

def ensure_file_in_context(file_path: str) -> bool:
    try:
        normalized_path = normalize_path(file_path)
        content = read_local_file(normalized_path)
        marker = f"User added file '{normalized_path}'"
        if not any(msg["role"] == "system" and marker in msg["content"] for msg in conversation_history):
            add_file_context_smartly(conversation_history, normalized_path, content)
        return True
    except (OSError, ValueError) as e:
        console.print(f"[red]✗ Error reading file for context '{file_path}': {e}[/red]")
        return False

def normalize_path(path_str: str) -> str:
    try:
        p = Path(path_str)
        
        # If path is absolute, use it as-is
        if p.is_absolute():
            if p.exists() or p.is_symlink(): 
                resolved_p = p.resolve(strict=True) 
            else:
                resolved_p = p.resolve()
        else:
            # For relative paths, resolve against base_dir instead of cwd
            base_path = base_dir / p
            if base_path.exists() or base_path.is_symlink():
                resolved_p = base_path.resolve(strict=True)
            else:
                resolved_p = base_path.resolve()
                
    except (FileNotFoundError, RuntimeError): 
        # Fallback: resolve relative to base_dir
        p = Path(path_str)
        if p.is_absolute():
            resolved_p = p.resolve()
        else:
            resolved_p = (base_dir / p).resolve()
    return str(resolved_p)

def create_gitignore():
    gitignore_path = Path(".gitignore")
    if gitignore_path.exists(): console.print("[yellow]⚠ .gitignore exists, skipping.[/yellow]"); return
    patterns = ["# Python", "__pycache__/", "*.pyc", "*.pyo", "*.pyd", ".Python", "env/", "venv/", ".venv", "ENV/", "*.egg-info/", "dist/", "build/", ".pytest_cache/", ".mypy_cache/", ".coverage", "htmlcov/", "", "# Env", ".env", ".env*.local", "!.env.example", "", "# IDE", ".vscode/", ".idea/", "*.swp", "*.swo", ".DS_Store", "", "# Logs", "*.log", "logs/", "", "# Temp", "*.tmp", "*.temp", "*.bak", "*.cache", "Thumbs.db", "desktop.ini", "", "# Node", "node_modules/", "npm-debug.log*", "yarn-debug.log*", "pnpm-lock.yaml", "package-lock.json", "", "# Local", "*.session", "*.checkpoint"]
    console.print("\n[bold bright_blue]📝 Creating .gitignore[/bold bright_blue]")
    if prompt_session.prompt("🔵 Add custom patterns? (y/n, default n): ", default="n").strip().lower() in ["y", "yes"]:
        console.print("[dim]Enter patterns (empty line to finish):[/dim]"); patterns.append("\n# Custom")
        while True: 
            pattern = prompt_session.prompt("  Pattern: ").strip()
            if pattern: patterns.append(pattern)
            else: break 
    try:
        with gitignore_path.open("w", encoding="utf-8") as f: f.write("\n".join(patterns) + "\n")
        console.print(f"[green]✓ Created .gitignore ({len(patterns)} patterns)[/green]")
        if git_context['enabled']: stage_file(str(gitignore_path))
    except OSError as e: console.print(f"[red]✗ Error creating .gitignore: {e}[/red]")

def stage_file(file_path_str: str) -> bool:
    if not git_context['enabled'] or git_context['skip_staging']: return False
    try:
        repo_root = Path.cwd()
        abs_file_path = Path(file_path_str).resolve() 
        rel_path = abs_file_path.relative_to(repo_root)
        result = subprocess.run(["git", "add", str(rel_path)], cwd=str(repo_root), capture_output=True, text=True, check=False)
        if result.returncode == 0: 
            console.print(f"[green dim]✓ Staged {rel_path}[/green dim]")
            return True
        else: 
            console.print(f"[yellow]⚠ Failed to stage {rel_path}: {result.stderr.strip()}[/yellow]")
            return False
    except ValueError: 
        console.print(f"[yellow]⚠ File {file_path_str} outside repo ({repo_root}), skipping staging[/yellow]")
        return False
    except Exception as e: 
        console.print(f"[red]✗ Error staging {file_path_str}: {e}[/red]")
        return False

def get_git_status_porcelain() -> Tuple[bool, List[Tuple[str, str]]]:
    if not git_context['enabled']: return False, []
    try:
        result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, cwd=str(Path.cwd()))
        if not result.stdout.strip(): return False, []
        changed_files = [(line[:2], line[3:]) for line in result.stdout.strip().split('\n') if line]
        return True, changed_files
    except subprocess.CalledProcessError as e: 
        console.print(f"[red]Error getting Git status: {e.stderr}[/red]")
        return False, []
    except FileNotFoundError: 
        console.print("[red]Git not found.[/red]")
        git_context['enabled'] = False
        return False, []

def user_commit_changes(message: str) -> bool:
    if not git_context['enabled']: 
        console.print("[yellow]Git not enabled.[/yellow]")
        return False
    try:
        add_all_res = subprocess.run(["git", "add", "-A"], cwd=str(Path.cwd()), capture_output=True, text=True)
        if add_all_res.returncode != 0: 
            console.print(f"[yellow]⚠ Failed to stage all: {add_all_res.stderr.strip()}[/yellow]")
        
        staged_check = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(Path.cwd()))
        if staged_check.returncode == 0: 
            console.print("[yellow]No changes staged for commit.[/yellow]")
            return False
        
        commit_res = subprocess.run(["git", "commit", "-m", message], cwd=str(Path.cwd()), capture_output=True, text=True)
        if commit_res.returncode == 0:
            console.print(f"[green]✓ Committed: \"{message}\"[/green]")
            log_info = subprocess.run(["git", "log", "--oneline", "-1"], cwd=str(Path.cwd()), capture_output=True, text=True).stdout.strip()
            if log_info: 
                console.print(f"[dim]Commit: {log_info}[/dim]")
            return True
        else: 
            console.print(f"[red]✗ Commit failed: {commit_res.stderr.strip()}[/red]")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        console.print(f"[red]✗ Git error: {e}[/red]")
        if isinstance(e, FileNotFoundError): 
            git_context['enabled'] = False
        return False

def initialize_git_repo_cmd() -> bool:
    if Path(".git").exists(): 
        console.print("[yellow]Git repo already exists.[/yellow]")
        git_context['enabled'] = True
        return True
    try:
        subprocess.run(["git", "init"], cwd=str(Path.cwd()), check=True, capture_output=True)
        git_context['enabled'] = True
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(Path.cwd()), capture_output=True, text=True)
        git_context['branch'] = branch_res.stdout.strip() if branch_res.returncode == 0 else "main"
        console.print(f"[green]✓ Initialized Git repo in {Path.cwd()}/.git/ (branch: {git_context['branch']})[/green]")
        if not Path(".gitignore").exists() and prompt_session.prompt("🔵 No .gitignore. Create one? (y/n, default y): ", default="y").strip().lower() in ["y", "yes"]: 
            create_gitignore()
        elif git_context['enabled'] and Path(".gitignore").exists(): 
            stage_file(".gitignore")
        if prompt_session.prompt(f"🔵 Initial commit? (y/n, default n): ", default="n").strip().lower() in ["y", "yes"]: 
            user_commit_changes("Initial commit")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        console.print(f"[red]✗ Failed to init Git: {e}[/red]")
        if isinstance(e, FileNotFoundError): 
            git_context['enabled'] = False
        return False

def create_git_branch_cmd(branch_name: str) -> bool:
    if not git_context['enabled']: 
        console.print("[yellow]Git not enabled.[/yellow]")
        return True
    if not branch_name: 
        console.print("[yellow]Branch name empty.[/yellow]")
        return True
    try:
        existing_raw = subprocess.run(["git", "branch", "--list", branch_name], cwd=str(Path.cwd()), capture_output=True, text=True)
        if existing_raw.stdout.strip():
            console.print(f"[yellow]Branch '{branch_name}' exists.[/yellow]")
            current_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True)
            if current_raw.stdout.strip() != branch_name and prompt_session.prompt(f"🔵 Switch to '{branch_name}'? (y/n, default y): ", default="y").strip().lower() in ["y", "yes"]:
                subprocess.run(["git", "checkout", branch_name], cwd=str(Path.cwd()), check=True, capture_output=True)
                git_context['branch'] = branch_name
                console.print(f"[green]✓ Switched to branch '{branch_name}'[/green]")
            return True
        subprocess.run(["git", "checkout", "-b", branch_name], cwd=str(Path.cwd()), check=True, capture_output=True)
        git_context['branch'] = branch_name
        console.print(f"[green]✓ Created & switched to new branch '{branch_name}'[/green]")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        console.print(f"[red]✗ Branch op failed: {e}[/red]")
        if isinstance(e, FileNotFoundError): 
            git_context['enabled'] = False
        return False

def show_git_status_cmd() -> bool:
    if not git_context['enabled']: 
        console.print("[yellow]Git not enabled.[/yellow]")
        return True
    has_changes, files = get_git_status_porcelain()
    branch_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True)
    branch_msg = f"On branch {branch_raw.stdout.strip()}" if branch_raw.returncode == 0 and branch_raw.stdout.strip() else "Not on any branch?"
    console.print(Panel(branch_msg, title="Git Status", border_style="blue", expand=False))
    if not has_changes: 
        console.print("[green]Working tree clean.[/green]")
        return True
    table = Table(show_header=True, header_style="bold bright_blue", border_style="blue")
    table.add_column("Sts", width=3); table.add_column("File Path"); table.add_column("Description", style="dim")
    s_map = {" M": (" M", "Mod (unstaged)"), "MM": ("MM", "Mod (staged&un)"), " A": (" A", "Add (unstaged)"), "AM": ("AM", "Add (staged&mod)"), "AD": ("AD", "Add (staged&del)"), " D": (" D", "Del (unstaged)"), "??": ("??", "Untracked"), "M ": ("M ", "Mod (staged)"), "A ": ("A ", "Add (staged)"), "D ": ("D ", "Del (staged)"), "R ": ("R ", "Ren (staged)"), "C ": ("C ", "Cop (staged)"), "U ": ("U ", "Unmerged")}
    staged, unstaged, untracked = False, False, False
    for code, filename in files:
        disp_code, desc = s_map.get(code, (code, "Unknown"))
        table.add_row(disp_code, filename, desc)
        if code == "??": untracked = True
        elif code.startswith(" "): unstaged = True
        else: staged = True
    console.print(table)
    if not staged and (unstaged or untracked): console.print("\n[yellow]No changes added to commit.[/yellow]")
    if staged: console.print("\n[green]Changes to be committed.[/green]")
    if unstaged: console.print("[yellow]Changes not staged for commit.[/yellow]")
    if untracked: console.print("[cyan]Untracked files present.[/cyan]")
    return True

# --- LLM Tool Handler Functions ---
def llm_git_init() -> str:
    if Path(".git").exists(): 
        git_context['enabled'] = True
        return "Git repository already exists."
    try:
        subprocess.run(["git", "init"], cwd=str(Path.cwd()), check=True, capture_output=True, text=True)
        git_context['enabled'] = True
        branch_res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(Path.cwd()), capture_output=True, text=True)
        git_context['branch'] = branch_res.stdout.strip() if branch_res.returncode == 0 else "main"
        if not Path(".gitignore").exists(): 
            create_gitignore()
        elif git_context['enabled']: 
            stage_file(".gitignore")
        return f"Git repository initialized successfully in {Path.cwd()}/.git/ (branch: {git_context['branch']})."

    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return f"Failed to initialize Git repository: {e}"

def llm_git_add(file_paths: List[str]) -> str:
    if not git_context['enabled']: return "Git not initialized."
    if not file_paths: return "No file paths to stage."
    staged_ok, failed_stage = [], []
    for fp_str in file_paths:
        try: 
            norm_fp = normalize_path(fp_str)
            if stage_file(norm_fp):
                staged_ok.append(norm_fp)
            else:
                failed_stage.append(norm_fp)
        except ValueError as e: 
            failed_stage.append(f"{fp_str} (path error: {e})")
        except Exception as e: 
            failed_stage.append(f"{fp_str} (error: {e})")
    res = []
    if staged_ok: res.append(f"Staged: {', '.join(Path(p).name for p in staged_ok)}")
    if failed_stage: res.append(f"Failed to stage: {', '.join(str(Path(p).name if isinstance(p,str) else p) for p in failed_stage)}")
    return ". ".join(res) + "." if res else "No files staged. Check paths."

def llm_git_commit(message: str) -> str:
    if not git_context['enabled']: return "Git not initialized."
    if not message: return "Commit message empty."
    try:
        staged_check = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=str(Path.cwd()))
        if staged_check.returncode == 0: return "No changes staged. Use git_add first."
        result = subprocess.run(["git", "commit", "-m", message], cwd=str(Path.cwd()), capture_output=True, text=True)
        if result.returncode == 0:
            info_raw = subprocess.run(["git", "log", "-1", "--pretty=%h %s"], cwd=str(Path.cwd()), capture_output=True, text=True).stdout.strip()
            return f"Committed. Commit: {info_raw}"
        return f"Failed to commit: {result.stderr.strip()}"
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return f"Git commit error: {e}"
    except Exception as e: 
        console.print_exception()
        return f"Unexpected commit error: {e}"

def llm_git_create_branch(branch_name: str) -> str:
    if not git_context['enabled']: return "Git not initialized."
    bn = branch_name.strip()
    if not bn: return "Branch name empty."
    try:
        exist_res = subprocess.run(["git", "rev-parse", "--verify", f"refs/heads/{bn}"], cwd=str(Path.cwd()), capture_output=True, text=True)
        if exist_res.returncode == 0:
            current_raw = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True)
            if current_raw.stdout.strip() == bn: return f"Already on branch '{bn}'."
            subprocess.run(["git", "checkout", bn], cwd=str(Path.cwd()), check=True, capture_output=True, text=True)
            git_context['branch'] = bn
            return f"Branch '{bn}' exists. Switched to it."
        subprocess.run(["git", "checkout", "-b", bn], cwd=str(Path.cwd()), check=True, capture_output=True, text=True)
        git_context['branch'] = bn
        return f"Created & switched to new branch '{bn}'."
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return f"Branch op failed for '{bn}': {e}"

def llm_git_status() -> str:
    if not git_context['enabled']: return "Git not initialized."
    try:
        branch_res = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True)
        branch_name = branch_res.stdout.strip() if branch_res.returncode == 0 and branch_res.stdout.strip() else "detached HEAD"
        has_changes, files = get_git_status_porcelain()
        if not has_changes: return f"On branch '{branch_name}'. Working tree clean."
        lines = [f"On branch '{branch_name}'."]
        staged, unstaged, untracked = [], [], []
        for code, filename in files:
            if code == "??": untracked.append(filename)
            elif code.startswith(" "): unstaged.append(f"{code.strip()} {filename}")
            else: staged.append(f"{code.strip()} {filename}")
        if staged: lines.extend(["\nChanges to be committed:"] + [f"  {s}" for s in staged])
        if unstaged: lines.extend(["\nChanges not staged for commit:"] + [f"  {s}" for s in unstaged])
        if untracked: lines.extend(["\nUntracked files:"] + [f"  {f}" for f in untracked])
        return "\n".join(lines)
    except (subprocess.CalledProcessError, FileNotFoundError) as e: 
        if isinstance(e, FileNotFoundError):
            git_context['enabled'] = False
        return f"Git status error: {e}"

# --- Function Call Execution ---
def execute_function_call_dict(tool_call_dict: Dict[str, Any]) -> str:
    func_name = "unknown_function"
    try:
        func_name = tool_call_dict["function"]["name"]
        args = json.loads(tool_call_dict["function"]["arguments"])
        if func_name == "read_file":
            norm_path = normalize_path(args["file_path"]); content = read_local_file(norm_path)
            return f"Content of file '{norm_path}':\n\n{content}"
        elif func_name == "read_multiple_files":
            results = []
            for fp in args["file_paths"]:
                try: 
                    norm_path = normalize_path(fp)
                    content = read_local_file(norm_path)
                    results.append(f"Content of '{norm_path}':\n\n{content}")
                except (OSError, ValueError) as e: 
                    results.append(f"Error reading '{fp}': {e}")
            return "\n\n" + "="*20 + " FILE SEP " + "="*20 + "\n\n".join(results)
        elif func_name == "create_file": 
            create_file(args["file_path"], args["content"])
            return f"File '{args['file_path']}' created/updated."
        elif func_name == "create_multiple_files":
            created, errors = [], []
            for f_info in args["files"]:
                try: 
                    create_file(f_info["path"], f_info["content"])
                    created.append(f_info["path"])
                except Exception as e: 
                    errors.append(f"Error creating {f_info.get('path','?path')}: {e}")
            res_parts = []
            if created: res_parts.append(f"Created/updated {len(created)} files: {', '.join(created)}")
            if errors: res_parts.append(f"Errors: {'; '.join(errors)}")
            return ". ".join(res_parts) if res_parts else "No files processed."
        elif func_name == "edit_file":
            fp = args["file_path"]
            if not ensure_file_in_context(fp): return f"Error: Could not read '{fp}' for editing."
            try: 
                apply_diff_edit(fp, args["original_snippet"], args["new_snippet"])
                return f"Edit attempt on '{fp}'. Check console."
            except Exception as e: 
                return f"Error during edit_file call for '{fp}': {e}."
        elif func_name == "git_init": return llm_git_init()
        elif func_name == "git_add": return llm_git_add(args.get("file_paths", []))
        elif func_name == "git_commit": return llm_git_commit(args.get("message", "Auto commit"))
        elif func_name == "git_create_branch": return llm_git_create_branch(args.get("branch_name", ""))
        elif func_name == "git_status": return llm_git_status()
        else: return f"Unknown LLM function: {func_name}"
    except json.JSONDecodeError as e: 
        console.print(f"[red]JSON Decode Error for {func_name}: {e}\nArgs: {tool_call_dict.get('function',{}).get('arguments','')}[/red]")
        return f"Error: Invalid JSON args for {func_name}."
    except KeyError as e: 
        console.print(f"[red]KeyError in {func_name}: Missing key {e}[/red]")
        return f"Error: Missing param for {func_name} (KeyError: {e})."
    except Exception as e: 
        console.print(f"[red]Unexpected Error in LLM func '{func_name}':[/red]")
        console.print_exception()
        return f"Unexpected error in {func_name}: {e}"

# --------------------------------------------------------------------------------
# 5. Conversation state
# --------------------------------------------------------------------------------
conversation_history: List[Dict[str, Any]] = [
    {"role": "system", "content": system_PROMPT}
]

# --------------------------------------------------------------------------------
# 6. Main loop (FIXED)
# --------------------------------------------------------------------------------
def main_loop():
    global conversation_history

    while True:
        try:
            prompt_indicator = f"🌳 {git_context['branch']} 🔵" if git_context['enabled'] and git_context['branch'] else "🔵"
            user_input = prompt_session.prompt(f"{prompt_indicator} You: ")
            
            if not user_input.strip(): continue

            if try_handle_add_command(user_input): continue
            if try_handle_commit_command(user_input): continue
            if try_handle_git_command(user_input): continue
            if try_handle_git_info_command(user_input): continue
            if try_handle_clear_command(user_input): continue
            if try_handle_folder_command(user_input): continue
            if try_handle_exit_command(user_input): continue
            if try_handle_help_command(user_input): continue
            
            conversation_history.append({"role": "user", "content": user_input})
            
            with console.status("[bold yellow]DeepSeek Engineer is thinking...[/bold yellow]", spinner="dots"):
                response_stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=conversation_history, # type: ignore 
                    tools=tools, # type: ignore 
                    tool_choice="auto",
                    stream=True
                )
            
            full_response_content = ""
            accumulated_tool_calls: List[Dict[str, Any]] = []

            console.print("[bold bright_magenta]🤖 DeepSeek Engineer:[/bold bright_magenta] ", end="")
            for chunk in response_stream:
                delta: ChoiceDelta = chunk.choices[0].delta
                if delta.content:
                    content_part = delta.content
                    console.print(content_part, end="", style="bright_magenta")
                    full_response_content += content_part
                
                if delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        idx = tool_call_chunk.index
                        while len(accumulated_tool_calls) <= idx:
                            accumulated_tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                        
                        current_tool_dict = accumulated_tool_calls[idx]
                        if tool_call_chunk.id: current_tool_dict["id"] = tool_call_chunk.id
                        if tool_call_chunk.function:
                            if tool_call_chunk.function.name: 
                                current_tool_dict["function"]["name"] = tool_call_chunk.function.name
                            if tool_call_chunk.function.arguments: 
                                current_tool_dict["function"]["arguments"] += tool_call_chunk.function.arguments
            console.print()

            # FIXED: Always add assistant message to maintain conversation flow
            assistant_message: Dict[str, Any] = {"role": "assistant"}
            
            # Always include content (even if empty) to maintain conversation flow
            assistant_message["content"] = full_response_content

            # Validate and add tool calls if any
            valid_tool_calls = validate_tool_calls(accumulated_tool_calls)
            if valid_tool_calls:
                assistant_message["tool_calls"] = valid_tool_calls
            
            # Always add the assistant message (maintains conversation flow)
            conversation_history.append(assistant_message)

            # Execute tool calls if any
            if valid_tool_calls:
                for tool_call_to_exec in valid_tool_calls: 
                    console.print(Panel(
                        f"[bold blue]Calling:[/bold blue] {tool_call_to_exec['function']['name']}\n"
                        f"[bold blue]Args:[/bold blue] {tool_call_to_exec['function']['arguments']}",
                        title="🛠️ Function Call", border_style="yellow", expand=False
                    ))
                    tool_output = execute_function_call_dict(tool_call_to_exec) 
                    console.print(Panel(tool_output, title=f"↪️ Output of {tool_call_to_exec['function']['name']}", border_style="green", expand=False))
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call_to_exec["id"],
                        "name": tool_call_to_exec["function"]["name"],
                        "content": tool_output 
                    })
            
            # FIXED: Smart truncation that preserves tool call sequences
            MAX_HISTORY_MESSAGES = 50  # Increased to accommodate tool sequences
            conversation_history = smart_truncate_history(conversation_history, MAX_HISTORY_MESSAGES)

        except KeyboardInterrupt: console.print("\n[yellow]⚠ Interrupted. Ctrl+D or /exit to quit.[/yellow]")
        except EOFError: console.print("[blue]👋 Goodbye! (EOF)[/blue]"); sys.exit(0)
        except Exception as e:
            console.print(f"\n[red]✗ Unexpected error in main loop:[/red]")
            console.print_exception(width=None, extra_lines=1, show_locals=True)

# --------------------------------------------------------------------------------
# 7. Entry point
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold bright_blue]🚀 DeepSeek Engineer Assistant[/bold bright_blue]\n"
        "[dim]Type /help for commands. Ctrl+C to interrupt, Ctrl+D or /exit to quit.[/dim]",
        border_style="bright_blue"
    ))
    
    if Path(".git").exists() and Path(".git").is_dir():
        git_context['enabled'] = True
        try:
            res = subprocess.run(["git", "branch", "--show-current"], cwd=str(Path.cwd()), capture_output=True, text=True, check=False)
            if res.returncode == 0 and res.stdout.strip(): git_context['branch'] = res.stdout.strip()
            else:
                init_branch_res = subprocess.run(["git", "config", "init.defaultBranch"], cwd=str(Path.cwd()), capture_output=True, text=True)
                git_context['branch'] = init_branch_res.stdout.strip() if init_branch_res.returncode == 0 and init_branch_res.stdout.strip() else "main"
        except FileNotFoundError: 
            console.print("[yellow]Git not found. Git features disabled.[/yellow]")
            git_context['enabled'] = False
        except Exception as e: 
            console.print(f"[yellow]Could not get Git branch: {e}.[/yellow]")
    
    main_loop()