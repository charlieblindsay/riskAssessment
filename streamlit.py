# Low hanging fruit:
# TODO: Improve few shot prompting examples so they don't parrot back input prompt
# TODO: Try using chain of thought prompt engineering for the mitigation prompt
# TODO: Try using Llama inference endpoint
# TODO: Try using Llama inference API but specify the number of tokens you want to receive
# TODO: Update question description in lambda feedback making it clear that 
# only one mitigation, one prevention and one 'how it harms' is specified

# Add option in RiskAssessment to specify whether prevention is misclassified as mitigation, 
# is not a suitable prevention, or mitigation is misclassified as prevention, or is not a suitable mitigation

from typing import Any, TypedDict
import numpy as np

import openai
import requests

from typing import Type

import os
from dotenv import load_dotenv

try:
    from .PromptInputs import *
except ImportError:
    from PromptInputs import *

class LLMCaller:
    def __init__(self):
        pass
         # NOTE: Don't need to pass self as input to calls of methods within a class 
         # as it is automatically passed in, i.e. it is not self.update_api_key_from_env_file(self) but:

    def update_api_key_from_env_file(self):
        pass

    def get_prompt_input(self, prompt_input: Type[PromptInput]):
        return prompt_input.generate_prompt()
    
    def get_JSON_output_from_API_call(self, prompt_input: Type[PromptInput]):
        pass

    def get_model_output(self):
        pass

class HuggingfaceLLMCaller(LLMCaller):
    def __init__(self, LLM_API_ENDPOINT):
        self.LLM_API_ENDPOINT = LLM_API_ENDPOINT
        self.update_api_key_from_env_file()
    
    def update_api_key_from_env_file(self):
        load_dotenv()
        self.HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]

class LLMWithGeneratedText(HuggingfaceLLMCaller):
    def __init__(self, LLM_API_ENDPOINT):
        super().__init__(LLM_API_ENDPOINT)
    
    def get_JSON_output_from_API_call(self, prompt_input: Type[PromptInput]):
        headers = {"Authorization": f"Bearer {self.HUGGINGFACE_API_KEY}"}
        prompt = prompt_input.generate_prompt()
        payload = {"inputs": prompt,
                   "options": {"wait_for_model": True}}
        return requests.post(self.LLM_API_ENDPOINT, 
                             headers=headers, 
                             json=payload).json()
    
    def get_model_output(self, prompt_input: Type[PromptInput]):
        LLM_output = self.get_JSON_output_from_API_call(prompt_input)
        return LLM_output[0]['generated_text']
    
class LLMWithCandidateLabels(HuggingfaceLLMCaller):
    def __init__(self, LLM_API_ENDPOINT):
        super().__init__(LLM_API_ENDPOINT)
    
    def get_JSON_output_from_API_call(self, prompt_input: Type[PromptInput]):
        headers = {"Authorization": f"Bearer {self.HUGGINGFACE_API_KEY}"}
        prompt = prompt_input.generate_prompt()
        payload = {"inputs": prompt,
                "parameters": {"candidate_labels": prompt_input.candidate_labels},
                "options": {"wait_for_model": True}}
        return requests.post(self.LLM_API_ENDPOINT, 
                             headers=headers, 
                             json=payload).json()

    def get_model_output(self, prompt_input: Type[PromptInput]):
        LLM_output = self.get_JSON_output_from_API_call(prompt_input)
        max_score_index = LLM_output['scores'].index(max(LLM_output['scores']))
        predicted_label = LLM_output['labels'][max_score_index]

        return predicted_label

class OpenAILLM(LLMCaller):
    def __init__(self):
        self.update_api_key_from_env_file()
        self.temperature = 0.5
        self.max_tokens = 300

    def update_api_key_from_env_file(self):
        load_dotenv()
        openai.api_key = st.secrets["OPENAI_API_KEY"]

    def get_JSON_output_from_API_call(self, prompt_input: Type[PromptInput]):

        prompt = self.get_prompt_input(prompt_input=prompt_input)
        
        messages = [{"role": "user", "content": prompt}]

        # TODO: Vary max_tokens based on prompt and test different temperatures.
        # NOTE: Lower temperature means more deterministic output.
        LLM_output = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                                  messages=messages, 
                                                  temperature=self.temperature, 
                                                  max_tokens=self.max_tokens)
        
        return LLM_output
    
    def get_model_output(self, prompt_input: Type[PromptInput]):
        LLM_output = self.get_JSON_output_from_API_call(prompt_input)
        return LLM_output.choices[0].message["content"]
    
import streamlit as st
from evaluation import evaluation_function

with st.form('risk_assessment'):
    activity = st.text_input('Activity', value='Fluids laboratory')
    hazard = st.text_input('Hazard', value="Ink spillage")
    how_it_harms = st.text_input('How it harms?', value="Serious eye damage")
    who_it_harms = st.text_input('Who it harms?', value="Students")
    uncontrolled_likelihood = st.text_input('Uncontrolled Likelihood', value='2')
    uncontrolled_severity = st.text_input('Uncontrolled Severity', value='2')
    uncontrolled_risk = st.text_input('Uncontrolled Risk', value='4')
    prevention = st.text_input('Prevention', value="Wear safety glasses")
    mitigation = st.text_input('Mitigation', value="Wash eyes with water")
    controlled_likelihood = st.text_input('Controlled Likelihood', value='1')
    controlled_severity = st.text_input('Controlled Severity', value='1')
    controlled_risk = st.text_input('Controlled Risk', value='1')
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        with st.spinner('Getting Risk Assessment Feedback...'):
            response = [activity, hazard, who_it_harms, how_it_harms, uncontrolled_likelihood, 
                        uncontrolled_severity, uncontrolled_risk, prevention, mitigation, 
                        controlled_likelihood, controlled_severity, controlled_risk]
            
            
            result = evaluation_function(response=response, answer='', params='')

            st.write(result)
            # st.write(result)