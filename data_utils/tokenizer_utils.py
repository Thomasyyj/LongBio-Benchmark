import tiktoken
from transformers import AutoTokenizer


class BiosTokenizer():
    def __init__(self, tokenizer_path):
        if 'gpt' in tokenizer_path:
            self.tokenizer = tiktoken.encoding_for_model(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True) if self.tokenizer_path else None
    
    def get_token_num(self, message):
        return self.tokenizer.encode(message)
    
    def get_token_num_from_chat(self, messages):
        num_tokens = 0
        tokens_per_message = 3
        tokens_per_name = 1
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(self.tokenizer.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
