# DeepSeek Engineer v2 🐋

## Overview

DeepSeek Engineer v2 is a powerful AI-powered coding assistant that provides an interactive terminal interface for seamless code development. It integrates with DeepSeek's advanced reasoning models to offer intelligent file operations, code analysis, and development assistance through natural conversation and function calling.

## 🚀 Latest Update: Enhanced Architecture & Model Switching

**Version 2.0** introduces significant upgrades from the original function calling architecture:
- **Dual Model Support**: Switch between DeepSeek Chat and DeepSeek Reasoner models
- **Enhanced Git Integration**: Full Git workflow support with automatic staging
- **Improved Context Management**: Smart conversation history and file context handling
- **Advanced Directory Management**: Configurable base directories for project work
- **Real-time reasoning visibility** with Chain of Thought (CoT) capabilities
- **Enhanced reliability** and better error handling

## Key Features

### 🧠 **AI Capabilities**
- **Elite Software Engineering**: Decades of experience across all programming domains
- **Dual Model Architecture**: 
  - **DeepSeek Chat** (💬): Fast, efficient responses for everyday coding tasks
  - **DeepSeek Reasoner** (🧠): Advanced reasoning with visible thought processes
- **Chain of Thought Reasoning**: Visible thought process before providing solutions (Reasoner mode)
- **Code Analysis & Discussion**: Expert-level insights and optimization suggestions
- **Intelligent Problem Solving**: Automatic file reading and context understanding

### 🛠️ **Function Calling Tools**
The AI can automatically execute these operations when needed:

#### **File Operations**
- `read_file(file_path: str)` - Read single file content with automatic path normalization
- `read_multiple_files(file_paths: List[str])` - Batch read multiple files efficiently
- `create_file(file_path: str, content: str)` - Create new files or overwrite existing ones
- `create_multiple_files(files: List[Dict])` - Create multiple files in a single operation
- `edit_file(file_path: str, original_snippet: str, new_snippet: str)` - Precise snippet-based file editing

#### **Git Operations** 
- `git_init()` - Initialize a new Git repository
- `git_add(file_paths: List[str])` - Stage files for commit
- `git_commit(message: str)` - Commit staged changes with a message
- `git_create_branch(branch_name: str)` - Create and switch to a new Git branch
- `git_status()` - Show current Git status

### 📁 **File Operations**

#### **Automatic File Reading (Recommended)**
The AI can automatically read files you mention:
```
You> Can you review the main.py file and suggest improvements?
→ AI automatically calls read_file("main.py")

You> Look at src/utils.py and tests/test_utils.py
→ AI automatically calls read_multiple_files(["src/utils.py", "tests/test_utils.py"])
```

#### **Manual Context Addition (Optional)**
For when you want to preload files into conversation context:
- **`/add path/to/file`** - Include single file in conversation context
- **`/add path/to/folder`** - Include entire directory (with smart filtering)

**Note**: The `/add` command is mainly useful when you want to provide extra context upfront. The AI can read files you mention in conversation automatically via function calls.

### 🎨 **Rich Terminal Interface**
- **Color-coded feedback** (green for success, red for errors, yellow for warnings)
- **Model indicators** (💬 for Chat, 🧠 for Reasoner)
- **Git branch display** in prompt when enabled
- **Real-time streaming** with visible reasoning process (Reasoner mode)
- **Structured panels** for function call execution and results
- **Progress indicators** for long operations

### 🔧 **Enhanced Commands**

#### **Model Management**
- **`/reasoner`** - Toggle between DeepSeek Chat and DeepSeek Reasoner models
- **`/r1`** - One-off call to DeepSeek Reasoner for complex reasoning tasks

#### **Git Workflow**
- **`/git init`** - Initialize Git repository with optional .gitignore creation
- **`/git status`** - Show detailed Git status with file staging information
- **`/git branch <name>`** - Create and switch to new branch
- **`/commit [message]`** - Stage all files and commit (prompts for message if not provided)

#### **Directory & Context Management**
- **`/folder [path]`** - Set base directory for file operations
- **`/folder reset`** - Reset to current working directory
- **`/clear-context`** - Clear conversation history while preserving system prompt
- **`/clear`** - Clear terminal screen

#### **Utility Commands**
- **`/help`** - Show comprehensive command reference
- **`/git-info`** - Display detailed Git capabilities
- **`/exit` or `/quit`** - Exit the application

### 🛡️ **Security & Safety**
- **Path normalization** and validation with configurable base directories
- **Directory traversal protection**
- **File size limits** (5MB per file)
- **Binary file detection** and exclusion
- **Smart context management** to prevent token overflow

## Getting Started

### Prerequisites
1. **DeepSeek API Key**: Get your API key from [DeepSeek Platform](https://platform.deepseek.com)
2. **Python 3.11+**: Required for optimal performance

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Doriandarko/deepseek-engineer.git
   cd deepseek-engineer
   ```

2. **Set up environment**:
   ```bash
   # Create .env file
   echo "DEEPSEEK_API_KEY=your_api_key_here" > .env
   ```

3. **Install dependencies** (choose one method):

   #### Using uv (recommended - faster)
   ```bash
   uv venv
   uv run deepseek-eng-r1.py
   ```

   #### Using pip
   ```bash
   pip install -r requirements.txt
   python3 deepseek-eng-r1.py
   ```

### Usage Examples

#### **Model Switching**
```
You> /reasoner
✓ Switched to DeepSeek Reasoner model 🧠

🧠 🔵 You> Explain the time complexity of quicksort

💭 Reasoning: The user is asking about quicksort time complexity. Let me think through this carefully...

🤖 DeepSeek Reasoner> Quicksort has different time complexities depending on the scenario...
```

#### **Natural Conversation with Automatic File Operations**
```
💬 🔵 You> Can you read the main.py file and create comprehensive tests?

🤖 DeepSeek Engineer> I'll read the main.py file first to understand its structure.

🛠️ Function Call
Calling: read_file
Args: {"file_path": "main.py"}

↪️ Output of read_file
Content of file 'main.py':
[file content shown]

Now I'll create comprehensive tests based on the code structure.

🛠️ Function Call  
Calling: create_file
Args: {"file_path": "test_main.py", "content": "..."}

✓ Created/updated file at 'test_main.py'
```

#### **Git Workflow Integration**
```
🌳 main 💬 🔵 You> /git branch feature/new-api

✓ Created & switched to new branch 'feature/new-api'

🌳 feature/new-api 💬 🔵 You> Create a new API endpoint for user management

🤖 DeepSeek Engineer> I'll create a new API endpoint for user management.

[AI creates files via function calls]

🌳 feature/new-api 💬 🔵 You> /commit "Add user management API endpoint"

✓ Committed: "Add user management API endpoint"
Commit: a1b2c3d Add user management API endpoint
```

#### **Advanced Project Analysis**
```
You> /add src/

✓ Added folder 'src/' to conversation.
📁 Added files: (15 files) utils.py, models.py, api.py...

You> /reasoner

You> Analyze this codebase and provide a comprehensive refactoring plan

💭 Reasoning: I need to analyze the entire codebase structure, identify patterns, potential issues, and areas for improvement...

🤖 DeepSeek Reasoner> After analyzing your codebase, I've identified several key areas for refactoring:

1. **Architectural Improvements**: The current structure could benefit from...
```

## Technical Details

### **Models**
- **DeepSeek-Chat**: Fast, efficient model for everyday coding tasks
- **DeepSeek-Reasoner (R1)**: Advanced reasoning model with visible Chain-of-Thought

### **Enhanced Architecture**
- **Smart Context Management**: Automatic file context with size limits and deduplication
- **Git Integration**: Automatic staging, branch tracking, and status monitoring  
- **Conversation History**: Smart truncation preserving tool call sequences
- **Error Recovery**: Graceful handling of API errors and file operation failures

### **Function Call Execution Flow**
1. **User Input** → Natural language request or command
2. **Model Selection** → Chat vs Reasoner based on current mode
3. **AI Processing** → Function calls executed automatically
4. **Real-time Feedback** → Operation status and results displayed
5. **Context Updates** → Conversation and file contexts maintained

## File Operations Comparison

| Method | When to Use | How It Works |
|--------|-------------|--------------|
| **Automatic Reading** | Most cases - just mention files | AI automatically calls `read_file()` when you reference files |
| **`/add` Command** | Preload context, bulk operations | Manually adds files to conversation context upfront |
| **Function Calls** | AI-driven operations | AI decides when to read/write files based on conversation |

**Recommendation**: Use natural conversation - the AI will automatically handle file operations. Use `/add` for bulk context loading.

## Script Variants

The repository includes two main scripts:

- **`deepseek-eng.py`** - Original function calling implementation
- **`deepseek-eng-r1.py`** - Enhanced version with model switching, Git integration, and advanced features

Choose `deepseek-eng-r1.py` for the full feature set including model switching and Git workflow support.

## Troubleshooting

### **Common Issues**

**API Key Not Found**
```bash
# Make sure .env file exists with your API key
echo "DEEPSEEK_API_KEY=your_key_here" > .env
```

**Import Errors**
```bash
# Install dependencies
uv sync  # or pip install -r requirements.txt
```

**File Permission Errors**
- Ensure you have write permissions in the working directory
- Use `/folder <path>` to set a different base directory
- Check file paths are correct and accessible

**Git Issues**
- Run `/git init` to initialize repository
- Ensure Git is installed and accessible
- Check current directory is within a Git repository

## Contributing

This project showcases DeepSeek reasoning model capabilities with enhanced architecture. Contributions are welcome!

### **Development Setup**
```bash
git clone https://github.com/Doriandarko/deepseek-engineer.git
cd deepseek-engineer
uv venv
uv sync
```

### **Run Enhanced Version**
```bash
# Run the enhanced application (preferred)
uv run deepseek-eng-r1.py
```

### **Run Original Version**
```bash
# Run the original application
uv run deepseek-eng.py
```

## Configuration

### **Environment Variables**
- `DEEPSEEK_API_KEY` - Your DeepSeek API key (required)

### **Runtime Configuration**
- **Model Selection**: Use `/reasoner` to toggle between models
- **Base Directory**: Use `/folder <path>` to set working directory
- **Git Integration**: Automatic when repository detected, manual via `/git init`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is experimental and developed for testing DeepSeek reasoning model capabilities.

---

> **Note**: This enhanced version (`deepseek-eng-r1.py`) includes model switching, Git integration, and advanced context management. The AI can automatically read files you mention in conversation, while commands like `/add` are available for bulk context loading. The original version (`deepseek-eng.py`) remains available for basic function calling workflows. Choose the version that best fits your development needs! 🚀