import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextStreamer
from huggingface_hub import login, snapshot_download
from dotenv import load_dotenv
from colorama import init, Fore, Style
import torch

init()
load_dotenv()
login(os.getenv('HF_TOKEN'))

def download_model(model_name, local_dir):
    snapshot_download(model_name, local_dir=local_dir)

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
    return tokenizer, model

class ColoredStreamer(TextStreamer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = False

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            self.token_cache.extend(self.tokenizer.encode("<|start|>assistant"))
            return
        elif self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            value = torch.cat([value[:-2], self.tokenizer.encode(" ", return_tensors='pt')[0], value[-2:]])

        if not self.message:
            if "<|channel|>analysis" == self.tokenizer.decode(self.token_cache[-2:]): 
                print(f"{Fore.CYAN}\n-------------THOUGHT-------------")
                self.message = True
                if self.decode_kwargs['skip_special_tokens']: self.token_cache = self.token_cache[:-3]
            elif "<|channel|>final" == self.tokenizer.decode(self.token_cache[-2:]): 
                print(f"{Fore.GREEN}\n-------------RESPONSE------------")
                self.message = True
                if self.decode_kwargs['skip_special_tokens']: self.token_cache = self.token_cache[:-3]
        elif self.message and (self.tokenizer.decode(value[-1:]) == "<|end|>" or self.tokenizer.decode(value[-1:]) == "<|return|>"):
            value = torch.cat([value, self.tokenizer.encode("\n", return_tensors='pt')[0]])
            self.message = False

        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)

def main():
    model_path = os.path.join('Models', 'GPT')

    tokenizer, model = load_model(model_path)
    pipe = pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        framework='pt',
        return_full_text=False,
        max_new_tokens=8_192,
        top_k=None,
        streamer=ColoredStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        ),
        temperature=0.6
    )

    os.system('cls')
    messages = []
    with open(os.path.join('prompts', 'system_prompt.txt'), 'r') as f: messages.append({'role': 'system', 'content': f.read()})
    
    print(f"{Fore.MAGENTA}Assistant:\n{Fore.GREEN}Hello, how can I help you today?\n")
    
    usr_msg = ""
    while usr_msg != "bye":
        usr_msg = input(f"{Fore.MAGENTA}User:\n{Fore.BLUE}"); print(Style.RESET_ALL)
        messages.append({'role': 'user', 'content': usr_msg})
        prompt = tokenizer.apply_chat_template(
            messages,
            reasoning_effort='meduim',
            model_identity="You are a helpful assistant.",
            tokenize=False,
            add_generation_prompt=True
        )
        print(f"{Fore.MAGENTA}Assistant:{Style.RESET_ALL}", end="")
        messages.append({'role': 'assistant', 'content': pipe(prompt)[0]['generated_text'].partition('assistantfinal')[-1]})

if __name__ == "__main__": main()