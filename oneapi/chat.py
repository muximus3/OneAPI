from rich import print
from rich.markdown import Markdown
from rich.rule import Rule
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.box import MINIMAL, ROUNDED, SIMPLE_HEAD
import re
import time
from openai.error import RateLimitError
import readline
import datetime
import json
from pathlib import Path
import sys 
import os
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from oneapi.utils import correct_zh_punctuation

class MessageBlock:

  def __init__(self):
    self.live = Live(auto_refresh=False, console=Console())
    self.live.start()
    self.content = ""

  def update_from_message(self, content):
    self.content += content
    if self.content:
      self.refresh()

  def end(self):
    self.refresh(cursor=False)
    self.live.stop()

  def refresh(self, cursor=True):
    # De-stylize any code blocks in markdown,
    # to differentiate from our Code Blocks
    # content = textify_markdown_code_blocks(self.content)
    content = self.content
    
    if cursor:
      content += "█"
      
    markdown = Markdown(content.strip())
    panel = Panel(markdown, box=MINIMAL, style="white on black")
    self.live.update(panel)
    self.live.refresh()


def textify_markdown_code_blocks(text):
  """
  To distinguish CodeBlocks from markdown code, we simply turn all markdown code
  (like '```python...') into text code blocks ('```text') which makes the code black and white.
  """
  replacement = "```text"
  lines = text.split('\n')
  inside_code_block = False

  for i in range(len(lines)):
    # If the line matches ``` followed by optional language specifier
    if re.match(r'^```(\w*)$', lines[i].strip()):
      inside_code_block = not inside_code_block

      # If we just entered a code block, replace the marker
      if inside_code_block:
        lines[i] = replacement

  return '\n'.join(lines)

class ChatAgent:
    
    def __init__(self, llm) -> None:
        self.messages = []
        self.temperature = 0.001
        self.local = False
        self.debug_mode = False
        self.model = "" 
        self.context_window = 2000 # For local models only
        self.max_tokens = 750 # For local models only
        self.llm_instance = None
        self.active_block = None
        self.tool = llm
        self.system_message = ""
    
    
    def reset(self):
        self.messages = []

    def load(self, messages):
        self.messages = messages

    def handle_undo(self):
        if len(self.messages) == 0:
            return
        message = self.messages.pop()
        print(Markdown(f"**Removed message:** `\"{message.get('content', message.get('value', ''))[:30]}...\"`"))
        print(self.tool._preprocess_claude_prompt(self.messages))
        print(Markdown(f"**Current messages:**"))

    def handle_clear(self):
        self.messages = []
        print(Markdown(f"**Cleared messages.**"))

    def handle_deep_clear(self):
        self.system_message = ''
        self.messages = []
        print(Markdown(f"**Cleared messages and system prompt.**"))

    def handle_save_messages(self):
        if len(self.messages) <= 1:
            return
        month = datetime.datetime.now().strftime("%Y%m")
        cace_dir = Path.home()/".cache"
        if not cace_dir.exists():
            cace_dir.mkdir(parents=True)
        cace_file_dir = cace_dir/f"history_cache_{month}.jsonl"
        with open(cace_file_dir, "a+" if cace_file_dir.is_file() else "w+") as f:
            format_data = {'id': str(time.time()), 'system_prompt': self.system_message, 'model': self.model, 'dataset_name': 'human', 'conversations': self.messages}
            f.write(json.dumps(format_data, ensure_ascii=False) + "\n")
        print(f'Save model response to file success! DIR: {cace_file_dir}')

    def handle_command(self, user_input):
        actions = {
            "clear": self.handle_clear,
            "d_clear": self.handle_deep_clear,
            "undo": self.handle_undo,
            "save": self.handle_save_messages
        }
        command = user_input[1:].strip()
        action = actions.get(command, None)
        if action:
            action()
        else:
            print(Markdown(f"**Unknown command:** `{command}`"))

    def end_active_block(self):
        if self.active_block:
            self.active_block.end()
            self.active_block = None

    def chat(self):
        if self.system_message:
            if self.active_block == None:
                # self.active_block = MessageBlock()
                # self.active_block.update_from_message(self.system_message)
                # self.active_block.end()
                print(Markdown(f"**System:** {self.system_message}"), end="\n\n")
        while True:
            try:
                user_input = input("> ").strip() 
            except EOFError:
                break
            except KeyboardInterrupt:
                self.handle_save_messages()
                break

            if not user_input.strip():
                continue
            try: 
                readline.add_history(user_input)
            except:
                pass

            if user_input.startswith((":", "：")):
                self.handle_command(user_input)
                continue
            
            self.messages.append({"role": "user", "content": user_input})

            try:
                self.respond()
            except KeyboardInterrupt:
                pass
            finally:
                self.end_active_block()

                
    
    def respond(self):
        token_counts = self.tool.count_tokens([m["content"] for m in self.messages])
        attempts = 16
        try:
            # print(self.system_message)
            # print(Markdown(json.dumps(self.messages, indent=2, ensure_ascii=False)))
            # print(self.tool._preprocess_claude_prompt(self.messages))
            response = self.tool.chat(self.messages, system=self.system_message, stream=True, model=self.model, temperature=self.temperature)
        except RateLimitError as e:
            print(Markdown(f"> We hit a rate limit. Cooling off for {attempts} seconds..."))
            time.sleep(attempts)
            try:
                response = self.tool.chat(self.messages, system=self.system_message, stream=True, model=self.model, temperature=self.temperature)
            except Exception as e:
                pass
        except Exception as e:
            print(Markdown(f"> {e}"))
            return
        plain_response = ""
        for chunk in response:
            if self.active_block == None:
                self.active_block = MessageBlock()
            self.active_block.update_from_message(chunk)
            plain_response += chunk
        if self.model.startswith('claude'):
            plain_response = correct_zh_punctuation(plain_response)
        self.messages.append({"role": "assistant", "content": plain_response})
        self.active_block.end()

            
