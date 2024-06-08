import openai
import requests
import anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import time

from typing import Type

import os
from dotenv import load_dotenv

try:
    from .PromptInputs import *
except ImportError:
    from PromptInputs import *

class LLMCaller:
    def __init__(self, name:str):
        pass

    def update_api_key_from_env_file(self):
        pass
    
    def get_JSON_output_from_API_call(self, prompt_input: Type[PromptInput]):
        pass

    def get_model_output(self):
        pass

class OpenAILLM(LLMCaller):
    def __init__(self, name, temperature):
        self.name = name
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.delay_between_requests = 0

    def update_api_key_from_env_file(self):
        load_dotenv()
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def get_JSON_output_from_API_call(self, prompt, max_tokens):
        messages = [{"role": "user", "content": prompt}]

        # TODO: Vary max_tokens based on prompt and test different temperatures.
        # NOTE: Lower temperature means more deterministic output.
        LLM_output = openai.ChatCompletion.create(model=self.name, 
                                            messages=messages, 
                                            temperature=self.temperature, 
                                            max_tokens=max_tokens)
        return LLM_output
    
    def get_model_output(self, prompt, max_tokens):
        time.sleep(self.delay_between_requests)
        LLM_output = self.get_JSON_output_from_API_call(prompt, max_tokens)
        return LLM_output.choices[0].message["content"]

class GPT_3_point_5_turbo(OpenAILLM):
    def __init__(self, temperature):
        self.name = 'gpt-3.5-turbo-1106'
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.delay_between_requests = 0

class GPT_4_turbo(OpenAILLM):
    def __init__(self, temperature):
        self.name = 'gpt-4-turbo'
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.delay_between_requests = 0

class AnthropicLLM(LLMCaller):
    def __init__(self, name: str, system_message: str, temperature: float):
        self.name = name
        self.system_message = system_message
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.delay_between_requests = 10

    def update_api_key_from_env_file(self):
        load_dotenv()
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

    def get_JSON_output_from_API_call(self, prompt:str, max_tokens:int):
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.name,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=self.system_message,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )

        return message
    
    def get_model_output(self, prompt, max_tokens):
        time.sleep(self.delay_between_requests)
        LLM_output = self.get_JSON_output_from_API_call(prompt, max_tokens)
        return LLM_output.content[0].text
    
class ClaudeHaikuLLM(AnthropicLLM):
    def __init__(self, system_message: str, temperature: float):
        self.name = 'claude-3-haiku-20240307'
        self.system_message = system_message
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.delay_between_requests = 0
    
class ClaudeSonnetLLM(AnthropicLLM):
    def __init__(self, system_message: str, temperature: float):
        self.name = 'claude-3-sonnet-20240229'
        self.system_message = system_message
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.delay_between_requests = 0

class ClaudeOpusLLM(AnthropicLLM):
    def __init__(self, system_message: str, temperature: float):
        self.name = 'claude-3-opus-20240229'
        self.system_message = system_message
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.delay_between_requests = 0

class MistralLLM(LLMCaller):
    def __init__(self, model, temperature):
        self.name = model
        self.temperature = temperature
        self.delay_between_requests = 0
        self.update_api_key_from_env_file()

    def update_api_key_from_env_file(self):
        load_dotenv()
        self.MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
    
    def get_JSON_output_from_API_call(self, prompt, max_tokens):
        client = MistralClient(api_key=self.MISTRAL_API_KEY)
        messages = [
            ChatMessage(role="user", content=prompt)
        ]

        chat_response = client.chat(
            model=self.name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=max_tokens
        )

        return chat_response
    
    def get_model_output(self, prompt, max_tokens):
        LLM_output = self.get_JSON_output_from_API_call(prompt, max_tokens)
        return LLM_output.choices[0].message.content
    
class Mixtral8x7B(MistralLLM):
    def __init__(self, temperature):
        self.model = 'open-mixtral-8x7b'
        self.name = self.model
        self.temperature = temperature
        self.delay_between_requests = 0
        self.update_api_key_from_env_file()

class Mixtral8x22B(MistralLLM):
    def __init__(self, temperature):
        self.model = 'open-mixtral-8x22b'
        self.name = self.model
        self.temperature = temperature
        self.delay_between_requests = 0
        self.update_api_key_from_env_file()

class MistralLarge(MistralLLM):
    def __init__(self, temperature):
        self.model = 'mistral-large-latest'
        self.name = self.model
        self.temperature = temperature
        self.delay_between_requests = 0
        self.update_api_key_from_env_file()

class HuggingfaceLLMCaller(LLMCaller):
    def __init__(self, name, LLM_API_ENDPOINT):
        self.name = name
        self.LLM_API_ENDPOINT = LLM_API_ENDPOINT
        self.update_api_key_from_env_file()
    
    def update_api_key_from_env_file(self):
        load_dotenv()
        self.HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

class HuggingfaceLLMWithGeneratedText(HuggingfaceLLMCaller):
    def __init__(self, name, LLM_API_ENDPOINT):
        super().__init__(name, LLM_API_ENDPOINT)
    
    def get_JSON_output_from_API_call(self, prompt):
        headers = {"Authorization": f"Bearer {self.HUGGINGFACE_API_KEY}"}
        payload = {"inputs": prompt,
                   "options": {"wait_for_model": True},
                   "parameters": {"return_full_text": False,
                                  "max_new_tokens": 250
                   }}
        return requests.post(self.LLM_API_ENDPOINT, 
                             headers=headers, 
                             json=payload).json()
    
    def get_model_output(self, prompt_input: Type[PromptInput]):
        LLM_output = self.get_JSON_output_from_API_call(prompt_input)
        return LLM_output[0]['generated_text']
    
class MixtralLLM(HuggingfaceLLMWithGeneratedText):
    def __init__(self):
        name = 'Mixtral-8x7B-Instruct-v0.1'
        LLM_API_ENDPOINT = 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1'
        
        super().__init__(name, LLM_API_ENDPOINT)

temperature = 0.1

LLM_dictionary = {
    # "Claude Opus": ClaudeOpusLLM(system_message="", temperature=temperature),
    # "Claude Sonnet": ClaudeSonnetLLM(system_message="", temperature=temperature),
    # "Claude Haiku": ClaudeHaikuLLM(system_message="", temperature=temperature),
    # "GPT-4 Turbo": GPT_4_turbo(temperature=temperature),
    "GPT-3.5 Turbo 1106": GPT_3_point_5_turbo(temperature=temperature),
    "Mistral Large": MistralLarge(temperature=temperature),
    "Mixtral 8x22B": Mixtral8x22B(temperature=temperature),
    "Mixtral 8x7B": Mixtral8x7B(temperature=temperature),
}