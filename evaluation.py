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
        self.HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

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
        openai.api_key = os.environ.get("OPENAI_API_KEY")

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

try:
    from RiskAssessment import RiskAssessment
    # from LLMCaller import LLMWithCandidateLabels, LLMWithGeneratedText
except:
    from .RiskAssessment import RiskAssessment
    # from .LLMCaller import LLMWithCandidateLabels, LLMWithGeneratedText, OpenAILLM

class Params(TypedDict):
    pass

class Result(TypedDict):
    input_check_feedback_message: str
    question_titles: list
    prompt_input_objects: list
    prompts: list
    prompt_outputs: list
    regex_matches: list
    shortform_feedbacks: list
    is_everything_correct: bool
    booleans_indicating_which_prompts_need_feedback: list
    controlled_risk: str
    uncontrolled_risk: str

def evaluation_function(response: Any, answer: Any, params: Any) -> Result:
    """
    Function used to evaluate a student response.
    ---
    The handler function passes three arguments to evaluation_function():

    - `response` which are the answers provided by the student.
    - `answer` which are the correct answers to compare against.
    - `params` which are any extra parameters that may be useful,
        e.g., error tolerances.

    The output of this function is what is returned as the API response
    and therefore must be JSON-encodable. It must also conform to the
    response schema.

    Any standard python library may be used, as well as any package
    available on pip (provided it is added to requirements.txt).

    The way you wish to structure you code (all in this function, or
    split into many) is entirely up to you. All that matters are the
    return types and that evaluation_function() is the main function used
    to output the evaluation response.
    """

    activity, hazard, who_it_harms, how_it_harms, uncontrolled_likelihood, uncontrolled_severity, uncontrolled_risk, prevention, mitigation, controlled_likelihood, controlled_severity, controlled_risk = np.array(response).flatten()

    RA = RiskAssessment(activity=activity, hazard=hazard, who_it_harms=who_it_harms, how_it_harms=how_it_harms,
                        uncontrolled_likelihood=uncontrolled_likelihood, uncontrolled_severity=uncontrolled_severity,
                        uncontrolled_risk=uncontrolled_risk, prevention=prevention, mitigation=mitigation,
                        controlled_likelihood=controlled_likelihood, controlled_severity=controlled_severity, controlled_risk=controlled_risk,
                        prevention_prompt_expected_output='prevention', mitigation_prompt_expected_output='mitigation')
    
    input_check_feedback_message = RA.get_input_check_feedback_message()
    controlled_risk = RA.check_controlled_risk()
    uncontrolled_risk = RA.check_uncontrolled_risk()

    if input_check_feedback_message != '':
        return Result(input_check_feedback_message=input_check_feedback_message,
                    question_titles=[], 
                        question='', 
                        prompt_input_objects=[],
                        prompts=[], 
                        prompt_outputs=[], 
                        regex_matches=[], 
                        shortform_feedbacks=[], 
                        is_everything_correct=False, 
                        booleans_indicating_which_prompts_need_feedback=[],
                        controlled_risk='',
                        uncontrolled_risk='')

    if input_check_feedback_message == '' and controlled_risk != 'correct' or uncontrolled_risk != 'correct':
        return Result(input_check_feedback_message=input_check_feedback_message,
                    question_titles=[], 
                        question='', 
                        prompt_input_objects=[],
                        prompts=[], 
                        prompt_outputs=[], 
                        regex_matches=[], 
                        shortform_feedbacks=[], 
                        is_everything_correct=False, 
                        booleans_indicating_which_prompts_need_feedback=[],
                        controlled_risk=controlled_risk,
                        uncontrolled_risk=uncontrolled_risk)
    
    if input_check_feedback_message == '' and controlled_risk == 'correct' and uncontrolled_risk == 'correct':
        LLM = OpenAILLM()
        question_titles = RA.get_list_of_question_titles()
        prompt_input_objects = RA.get_list_of_prompt_input_objects()
        prompts = RA.get_list_of_prompts()
        prompt_outputs = RA.get_list_of_prompt_outputs(LLM)
        regex_matches = RA.get_list_of_regex_matches(prompt_outputs)
        shortform_feedbacks = RA.get_list_of_shortform_feedback_from_regex_matches(regex_matches)
        is_everything_correct = RA.are_all_prompt_outputs_correct(prompt_outputs) and RA.are_all_multiplications_correct()
        booleans_indicating_which_prompts_need_feedback = RA.get_booleans_indicating_which_prompts_need_feedback(regex_matches)

        
        return Result(input_check_feedback_message=input_check_feedback_message,
                    question_titles=question_titles, 
                        prompt_input_objects=prompt_input_objects,
                        prompts=prompts, 
                        prompt_outputs=prompt_outputs, 
                        regex_matches=regex_matches, 
                        shortform_feedbacks=shortform_feedbacks, 
                        is_everything_correct=is_everything_correct, 
                        booleans_indicating_which_prompts_need_feedback=booleans_indicating_which_prompts_need_feedback,
                        controlled_risk=controlled_risk,
                        uncontrolled_risk=uncontrolled_risk)