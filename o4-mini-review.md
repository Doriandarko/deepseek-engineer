Here are a few high-level design ideas and concrete refactorings to make your context management more robust, simpler to reason about, and—critically—much easier to test and maintain.

Separate “file contexts” from “chat history”
Right now you interleave “User added file…” system messages into conversation_history alongside user/assistant/tool turns. That forces your truncation logic to scan the entire history for file markers, remove duplicates, sort by size/recency, etc. Instead, keep:

• a simple list of user/assistant/tool messages
• a separate file_contexts: List[FileContext] structure

At LLM-call time you merge them (up to your token limit). This cleanly separates concerns.

Use a ContextManager class
Encapsulate everything about token estimation, message queues, file contexts and truncation into one class:

import tiktoken  # for real token counts
from collections import deque
from typing import Deque, List, Optional
from dataclasses import dataclass, field

@dataclass
class FileContext:
  path: str
  content: str
  tokens: int

@dataclass
class Message:
  role: str
  content: str
  tokens: int

class ContextManager:
  def __init__(self,
               system_prompt: str,
               max_tokens: int = 120_000,
               max_file_contexts: int = 5,
               warn_threshold: float = 0.8,
               critical_threshold: float = 0.9):
    self.system_prompt = system_prompt
    self.max_tokens = max_tokens
    self.warn_threshold = warn_threshold
    self.critical_threshold = critical_threshold

    self.messages: Deque[Message] = deque()       # user/assistant/tool history
    self.file_contexts: List[FileContext] = []    # separate
    self.enc = tiktoken.encoding_for_model(DEFAULT_MODEL)

  def _count_tokens(self, text: str) -> int:
    return len(self.enc.encode(text))

  def add_message(self, role: str, content: str):
    self.messages.append( Message(role, content, self._count_tokens(content)) )

  def add_file(self, path: str, content: str) -> bool:
    tok = self._count_tokens(content)
    if tok > self.max_tokens // 10:
      console.print(f"[yellow]File too large ({tok} tokens), skipping[/yellow]")
      return False
    # remove existing entry for path
    self.file_contexts = [fc for fc in self.file_contexts if fc.path != path]
    self.file_contexts.append(FileContext(path, content, tok))
    # enforce max files
    while len(self.file_contexts) > self.max_file_contexts:
      removed = self.file_contexts.pop(0)
      console.print(f"[dim]Dropped oldest file context {removed.path}[/dim]")
    return True

  def usage(self) -> dict:
    total = sum(m.tokens for m in self.messages) + sum(fc.tokens for fc in self.file_contexts)
    return {
      "total_tokens": total,
      "pct": total / self.max_tokens,
      "status": ("green","yellow","red")[
        int(total > self.max_tokens * self.warn_threshold) +
        int(total > self.max_tokens * self.critical_threshold)
      ]
    }

  def build_payload(self, extra_user: Optional[str]=None) -> List[dict]:
    """
    Merge system prompt, file contexts, and as many of the last messages
    as will fit under max_tokens. Returns a list of dicts for LLM.
    """
    chunks: List[dict] = []
    # 1) system prompt
    chunks.append({"role":"system","content":self.system_prompt})

    # 2) file contexts
    fc_tokens = 0
    for fc in self.file_contexts:
      if fc_tokens + fc.tokens > self.max_tokens * 0.1:  # reserve 10%
        break
      chunks.append({"role":"system", "content":f"User added file '{fc.path}':\n\n{fc.content}"})
      fc_tokens += fc.tokens

    # 3) conversation, from the back
    conv_tokens = sum(m.tokens for m in chunks)
    for msg in reversed(self.messages):
      if conv_tokens + msg.tokens > self.max_tokens: break
      chunks.insert(len(chunks), {"role":msg.role,"content":msg.content})
      conv_tokens += msg.tokens

    # 4) optionally extra_user
    if extra_user:
      tok = self._count_tokens(extra_user)
      if conv_tokens + tok <= self.max_tokens:
        chunks.append({"role":"user","content":extra_user})

    return chunks
• Pros: Tests can target add_file, add_message, usage(), and build_payload() in isolation.
• Accurate counts: using tiktoken instead of heuristics.
• Simplified truncation: we drop oldest file contexts, then oldest messages, no nested loops chasing tool sequences.

Summaries for big files
If a file is large, instead of injecting full content, you can:

a) chunk it and only keep the first/last N lines, or
b) call the LLM once to “summarize” the file and store that summary as your content in FileContext.

def summarize_and_add(self, path:str, content:str):
  summary = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[
      {"role":"system", "content":"You’re a summarizer."},
      {"role":"user",   "content":f"Summarize this file:\n\n{content}"}
    ]
  ).choices[0].message.content
  return self.add_file(path, summary)
In your main loop, replace all of your ad-hoc context logic with:

cm = ContextManager(SYSTEM_PROMPT)
# … when user or assistant/tool messages happen:
cm.add_message(role, content)
# … when they do “/add foo.py”:
text = read_local_file(foo)
cm.add_file(foo, text)
# … when you’re about to call OpenAI:
payload = cm.build_payload(extra_user=user_input)
response = client.chat.completions.create(
  model=cm.current_model,
  messages=payload,
  tools=tools, tool_choice="auto", stream=True
)
Benefits of this refactoring

• Single responsibility: ContextManager only cares about context.
• Predictable: no more magic 0.1 or 0.25 of targets sprinkled through code, all of that lives in one place.
• Testable: you can write unit tests for every aspect of ContextManager.
• Maintainable: when your token‐budget changes, you only tune a couple of parameters in one constructor.

That should give you a much cleaner, more reliable foundation for managing both your chat history and file contexts—and it will save you from wrestling with complex truncation loops in the hot path.