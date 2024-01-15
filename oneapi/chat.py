from rich import print
from rich.markdown import Markdown
from rich.rule import Rule
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.markdown import Markdown
from rich.box import MINIMAL, ROUNDED, SIMPLE_HEAD
from rich.padding import Padding
import time
import readline
import datetime
import json
from openai import RateLimitError
from pathlib import Path
import traceback
import sys 
import os
sys.path.append(os.path.normpath(f"{os.path.dirname(os.path.abspath(__file__))}/.."))
from oneapi.utils import correct_zh_punctuation, find_files_unrecu, load_jsonl, map_conversations_format

class MessageBlock:

  def __init__(self, consule=Console()):
    self.console = consule
    self.content = []

  def __enter__(self):
    self.live = Live(auto_refresh=False, console=self.console)
    self.live.start()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.refresh(cursor=False)
    self.live.stop()

  def add_message(self, content):
    self.content.append(content)
    self.refresh()
      
  def refresh(self, cursor=True):
        text = ''.join(self.content)
        if cursor:
            text += ' █'
        markdown = Markdown(text)
        panel = Padding(Panel(markdown, box=MINIMAL, style="bright_white on gray7"), (1, 0, 1, 0))
        # panel.expand = True
        self.live.update(panel, refresh=True)


class ChatAgent:
    
    def __init__(self, llm) -> None:
        self.messages = []
        self.temperature = 0.1
        self.model = "" 
        self.max_tokens = 1024 # For local models only
        self.llm = llm
        self.ask_for_system_prompt = False
        self.system_message = ""
        self.consule = Console()
        month = datetime.datetime.now().strftime("%Y%m")
        self.cace_dir = Path.home()/".cache"
        if not self.cace_dir.exists():
            self.cace_dir.mkdir(parents=True)
        self.cace_file_dir = self.cace_dir/f"history_cache_{month}.jsonl" 
    
    def handle_undo(self):
        if len(self.messages) == 0:
            return
        message = self.messages.pop()
        print(Markdown(f"**Removed message:** `\"{message.get('content', message.get('value', ''))[:30]}...\"`"), end="\n\n")
        if len(self.messages) == 0:
            return
        print(Markdown(f"**Current message:** `\"{self.messages[-1].get('content', message.get('value', ''))[:30]}...\"`"), end="\n\n")
        if self.messages[-1]['role'] == 'user':
            print(Markdown(f"Press `Enter` to regenerate answer"), end="\n\n")

    def handle_clear(self):
        self.consule.clear()
        self.messages = []
        print(Markdown(f"**Cleared messages.**"))
        self.replay_messages()

    def handle_deep_clear(self):
        self.consule.clear()
        self.system_message = ''
        self.messages = []
        print(Markdown(f"**Cleared messages and system prompt.**"))
        self.set_system_prompt()

    def handle_save_messages(self):
        if len(self.messages) <= 1 or self.messages[-1]['role'] != 'assistant':
            return
        with open(self.cace_file_dir, "a+" if self.cace_file_dir.is_file() else "w+") as f:
            format_data = {'id': str(time.time()), 'system_prompt': self.system_message, 'model': self.model, 'dataset_name': 'human', 'conversations': self.messages}
            f.write(json.dumps(format_data, ensure_ascii=False) + "\n")
        print(f'Save model response to file success! DIR: {self.cace_file_dir}')
    


    def handle_load_history(self, idx: int=-1):
        idx = int(idx)
        if not self.cace_file_dir.is_file():
            files = find_files_unrecu(self.cace_dir, 'history_cache_*.jsonl')
            if len(files) == 0:
                print(f'No history cache file found in {self.cace_dir}')
                return
            files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            latest_file = files[0]
            print(f'Load history cache file {latest_file}')
        else:
            latest_file = self.cace_file_dir
        sessions = load_jsonl(latest_file)
        if idx > len(sessions) - 1:
            idx = -1
        self.system_message = sessions[idx]['system_prompt']
        self.messages = map_conversations_format(sessions[idx]['conversations'])
        self.replay_messages() 

        

    def handle_command(self, user_input):
        actions = {
            "clear": self.handle_clear,
            "d_clear": self.handle_deep_clear,
            "undo": self.handle_undo,
            "save": self.handle_save_messages,
            "load": self.handle_load_history,
            "system": self.set_system_prompt,
        }
        input_info = user_input[1:].strip().split()
        command = input_info[0]
        addtion_info = input_info[1] if len(input_info) > 1 else None
        action = actions.get(command, None)
        if action:
            if addtion_info:
                action(addtion_info)
            else:
                action()
        else:
            print(Markdown(f"**Unknown command:** `{command}`"))

    def replay_messages(self):
        if self.system_message:
            print(Markdown(f"**System:** {self.system_message}"), end="\n\n")
        for message in self.messages:
            if message['role'] == 'user':
                print(Markdown(f"> {message['content']}"))
            else:
                with MessageBlock(consule=self.consule) as block:
                    block.add_message(message['content'])
    def print_welcome(self):
        print(Rule(style='white', end="\n\n"))
        print(Markdown(f"Using **Model:** `{self.model}`"), end="\n\n")
        print(Rule(style='white', title="Start chat", end="\n\n"))
        if self.system_message:
            print(Markdown(f"**System:** {self.system_message}"), end="\n\n")
    
    def set_system_prompt(self):
        print(Markdown(f"Set `system prompt`, press enter to skip: "))
        system_message = input("> ")
        self.system_message = system_message
        self.print_welcome()

    def chat(self):
        if self.ask_for_system_prompt:
            self.set_system_prompt()
        else:
            self.print_welcome()
        while True:
            time.sleep(0.05)
            try:
                user_input = input("> ").strip() 
            except EOFError:
                break
            except KeyboardInterrupt:
                self.handle_save_messages()
                break

            if not user_input.strip():
                # regenerate
                if len(self.messages) > 0 and self.messages[-1]['role'] == 'user':
                    try:
                        self.respond()    
                    except KeyboardInterrupt:
                        pass
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

                
    
    def respond(self):
        attempts = 16
        try:
            # print(self.system_message)
            # print(Markdown(json.dumps(self.messages, indent=2, ensure_ascii=False)))
            # print(self.tool._preprocess_claude_prompt(self.messages))
            response = self.llm(self.messages, system=self.system_message, stream=True, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
        except RateLimitError as e:
            print(Markdown(f"> We hit a rate limit. Cooling off for {attempts} seconds..."))
            time.sleep(attempts)
            try:
                response = self.llm(self.messages, system=self.system_message, stream=True, model=self.model, temperature=self.temperature, max_tokens=self.max_tokens)
            except Exception as e:
                pass
        except Exception as e:
            traceback.print_exc()
            print(Markdown(f"> {e}"))
            return
        plain_response = ""
        with MessageBlock(consule=self.consule) as block:
            for chunk in response:
                time.sleep(0.01)
                block.add_message(chunk)
                plain_response += chunk
        if self.model.startswith('claude'):
            plain_response = correct_zh_punctuation(plain_response)
        self.messages.append({"role": "assistant", "content": plain_response})

            
