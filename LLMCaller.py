import streamlit as st
import pandas as pd

import time
import anthropic
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from typing import Type
from PromptInputs import PromptInput

import os
from dotenv import load_dotenv
import openai

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
    def __init__(self):
        # TODO: Would it be better to use the gpt-instruct model?
        self.name = 'gpt-3.5-turbo'
        self.update_api_key_from_env_file()
        self.temperature = 0.1
        self.max_tokens = 300
        self.delay_between_requests = 0

    def update_api_key_from_env_file(self):
        load_dotenv()
        openai.api_key = os.environ.get("OPENAI_API_KEY")

    def get_JSON_output_from_API_call(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        # TODO: Vary max_tokens based on prompt and test different temperatures.
        # NOTE: Lower temperature means more deterministic output.
        LLM_output = openai.ChatCompletion.create(model=self.name, 
                                            messages=messages, 
                                            temperature=self.temperature, 
                                            max_tokens=self.max_tokens)
        return LLM_output
    
    def get_model_output(self, prompt):
        time.sleep(self.delay_between_requests)
        LLM_output = self.get_JSON_output_from_API_call(prompt)
        return LLM_output.choices[0].message["content"]

class AnthropicLLM(LLMCaller):
    def __init__(self, name: str, system_message: str, temperature: float, max_tokens: int):
        self.name = 'claude-3-sonnet-20240229'
        self.system_message = system_message
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay_between_requests = 10

    def update_api_key_from_env_file(self):
        load_dotenv()
        self.ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

    def get_JSON_output_from_API_call(self, prompt:str ):
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=self.name,
            max_tokens=self.max_tokens,
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
    
    def get_model_output(self, prompt):
        time.sleep(self.delay_between_requests)
        LLM_output = self.get_JSON_output_from_API_call(prompt)
        return LLM_output.content[0].text
    
class ClaudeSonnetLLM(AnthropicLLM):
    def __init__(self, system_message: str, temperature: float, max_tokens: int):
        self.name = 'claude-3-sonnet-20240229'
        self.system_message = system_message
        self.update_api_key_from_env_file()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay_between_requests = 5

class MistralLLM(LLMCaller):
    def __init__(self, model, temperature, max_tokens):
        self.name = model
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.delay_between_requests = 0
        self.update_api_key_from_env_file()

    def update_api_key_from_env_file(self):
        load_dotenv()
        self.MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
    
    def get_JSON_output_from_API_call(self, prompt):
        client = MistralClient(api_key=self.MISTRAL_API_KEY)
        messages = [
            ChatMessage(role="user", content=prompt)
        ]

        chat_response = client.chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        return chat_response
    
    def get_model_output(self, prompt):
        time.sleep(self.delay_between_requests)
        LLM_output = self.get_JSON_output_from_API_call(prompt)
        return LLM_output.choices[0].message.content

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