# DeepSeek Engineer v2 🐋

## Overview

DeepSeek Engineer v2 is a powerful AI-powered coding assistant that provides an interactive terminal interface for seamless code development. It integrates with DeepSeek's advanced reasoning models to offer intelligent file operations, code analysis, and development assistance through natural conversation and function calling.

## 🚀 Latest Update: Function Calling Architecture

**Version 2.0** introduces a big upgrade from structured JSON output to native function calling, providing:
- **Natural conversations** with the AI without rigid response formats.
- **Automatic file operations** through intelligent function calls.
- **Real-time reasoning visibility** with Chain of Thought (CoT) capabilities.
- **Enhanced reliability** and better error handling.

## Key Features

### 🧠 **AI Capabilities**
- **Elite Software Engineering**: Decades of experience across all programming domains.
- **Chain of Thought Reasoning**: Visible thought process before providing solutions.
- **Code Analysis & Discussion**: Expert-level insights and optimization suggestions.
- **Intelligent Problem Solving**: Automatic file reading and context understanding.

### 🛠️ **Function Calling Tools**
The AI can automatically execute these operations when needed:

#### `read_file(file_path: str)`
- Read single file content with automatic path normalization.
- Built-in error handling for missing or inaccessible files.

#### `read_multiple_files(file_paths: List[str])`
- Batch read multiple files efficiently.
- Formatted output with clear file separators.

#### `create_file(file_path: str, content: str)`
- Create new files or overwrite existing ones.
- Automatic directory creation and safety checks.

#### `edit_file(file_path: str, original_snippet: str, new_snippet: str)`
- Precise snippet-based file editing.
- Safe replacement with exact matching.

### 📁 **File Operations**

#### **Automatic File Reading (Recommended)**
The AI can automatically read files you mention:
```
You> Can you review the main.py file and suggest improvements?
→ AI automatically calls read_file("main.py")
```

#### **Manual Context Addition (Optional)**
For when you want to preload files into conversation context:
- **`/add path/to/file`** - Include single file in conversation context.
- **`/add path/to/folder`** - Include entire directory (with smart filtering).

### 🎨 **Rich Terminal Interface**
- **Color-coded feedback** (green for success, red for errors, yellow for warnings).
- **Real-time streaming** with visible reasoning process.
- **Structured tables** for diff previews.
- **Progress indicators** for long operations.

## Getting Started

### Prerequisites
1. **DeepSeek API Key**: Get your API key from [DeepSeek Platform](https://platform.deepseek.com).
2. **Python 3.11+**: Required for optimal performance.

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

   #### Using `uv` (recommended - faster):
   ```bash
   uv venv
   uv run deepseek-eng.py
   ```

   #### Using `pip`:
   ```bash
   pip install -r requirements.txt
   python deepseek-eng.py
   ```

### Usage Examples

#### **Natural Conversation with Automatic File Operations**
```
You> Can you read the main.py file and create a test file for it?

💭 Reasoning: I need to first read the main.py file to understand its structure...

🤖 Assistant> I'll read the main.py file first to understand its structure.
⚡ Executing 1 function call(s)...
→ read_file
✓ Read file 'main.py'

🔄 Processing results...
Now I'll create comprehensive tests based on the code structure I found.
⚡ Executing 1 function call(s)...
→ create_file
✓ Created/updated file at 'test_main.py'

I've analyzed main.py and created comprehensive tests covering all the main functions...
```

## Contributing

Contributions are welcome! Please fork the [official repository](https://github.com/Doriandarko/deepseek-engineer.git) and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

📝 **Note**: This is an experimental project developed to explore the capabilities of DeepSeek's reasoning model with function calling. Use responsibly and enjoy the enhanced AI pair programming experience! 🚀