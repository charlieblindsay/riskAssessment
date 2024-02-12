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
    is_correct: bool
    feedback: str

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
                        prevention_prompt_expected_output='prevention', mitigation_prompt_expected_output='mitigation',
                        prevention_protected_clothing_expected_output=False,
                        mitigation_protected_clothing_expected_output=False,
                        prevention_first_aid_expected_output=False,
                        mitigation_first_aid_expected_output=False,
                        )
    
    input_check_feedback_message = RA.get_input_check_feedback_message()
    controlled_risk = RA.check_controlled_risk()
    uncontrolled_risk = RA.check_uncontrolled_risk()

    if input_check_feedback_message != '':
        return Result(is_correct=False,
                    feedback=f'Feedback: \n\n {input_check_feedback_message}')
    
    if input_check_feedback_message == '' and controlled_risk != 'correct' or uncontrolled_risk != 'correct':
        return Result(is_correct=False,
                      feedback=f'Feedback: \n\n Uncontrolled risk is: {controlled_risk}\n\nUncontrolled risk is {uncontrolled_risk}')
    
    else:
        LLM = OpenAILLM()

        feedback_for_incorrect_answers = '# Feedback for Incorrect Answers\n'
        feedback_for_correct_answers = '# Feedback for Correct Answers\n'

        is_everything_correct = True

        first_3_prompt_input_objects = RA.get_list_of_prompt_input_objects_for_first_3_prompts()

        for prompt_input_object in first_3_prompt_input_objects:
            prompt_output, pattern = RA.get_prompt_output_and_pattern_matched(prompt_input_object, LLM)
            shortform_feedback = RA.get_shortform_feedback_from_regex_match(prompt_input_object, pattern)

            field = prompt_input_object.get_field_checked()

            longform_feedback = prompt_input_object.get_longform_feedback(prompt_output=prompt_output)
            
            feedback_header_to_add = f''' 
            \n\n\n## Feedback for Input: {field}\n\n\n
            '''

            feedback_to_add = f'''
            \n\n\n\n##### Feedback: {shortform_feedback}\n\n\n\n
            \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n'''
            
            if pattern in prompt_input_object.labels_indicating_correct_input:
                feedback_for_correct_answers += feedback_header_to_add
                feedback_for_correct_answers += feedback_to_add + '\n\n'
            
            else:
                is_everything_correct = False

                feedback_to_add += f'''\n\n\n\n##### Recommendation: Please look at the definition of the {field} input field{'s' if field in ['Hazard & How it harms', 'Prevention and Mitigation'] else ''} and the example risk assessment for assistance.\n\n\n\n'''

                feedback_for_incorrect_answers += feedback_header_to_add
                feedback_for_incorrect_answers += feedback_to_add

                break
        
        # PREVENTION CHECKS
        feedback_header = f'''\n\n\n## Feedback for Input: Prevention\n\n\n'''

        if is_everything_correct == True:
            prevention_protective_clothing_prompt_input = RA.get_prevention_protective_clothing_input()
            prevention_protective_clothing_prompt_output, prevention_protective_clothing_pattern = RA.get_prompt_output_and_pattern_matched(prevention_protective_clothing_prompt_input, LLM)
            
            # Indicating that the prevention is a protective clothing so is actually a mitigation
            if prevention_protective_clothing_pattern == True:
                shortform_feedback = prevention_protective_clothing_prompt_input.get_shortform_feedback()
                longform_feedback = prevention_protective_clothing_prompt_input.get_longform_feedback()

                feedback_for_incorrect_answers += f'''
                {feedback_header}
                \n\n\n\n##### Feedback: {shortform_feedback.negative_feedback}\n\n\n\n
                \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n
                \n\n\n\n##### Recommendation: Please look at the definition of a Prevention and Mitigation for assistance.\n\n\n\n'''

                is_everything_correct = False
            
            # Indicating that the prevention is not a protective clothing
            else:
                prevention_first_aid_prompt_input = RA.get_prevention_first_aid_input()
                prevention_first_aid_prompt_output, prevention_first_aid_pattern = RA.get_prompt_output_and_pattern_matched(prevention_first_aid_prompt_input, LLM)

                # Indicating that the prevention is an example of first aid so is a mitigation
                if prevention_first_aid_pattern == True:
                    shortform_feedback = prevention_first_aid_prompt_input.get_shortform_feedback()
                    longform_feedback = prevention_first_aid_prompt_input.get_longform_feedback()

                    feedback_for_incorrect_answers += f'''
                    {feedback_header}
                    \n\n\n\n##### Feedback: {shortform_feedback.negative_feedback}\n\n\n\n
                    \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n
                    \n\n\n\n##### Recommendation: Please look at the definition of a Prevention and Mitigation for assistance.\n\n\n\n'''

                    is_everything_correct = False
                
                # Indicating that the prevention is neither a protective clothing nor an example of first aid
                # This checks whether the inputted prevention is a prevention or a mitigation 
                else:
                    prevention_prompt_input = RA.get_prevention_input()
                    prevention_prompt_output, prevention_pattern = RA.get_prompt_output_and_pattern_matched(prevention_prompt_input, LLM)

                    shortform_feedback_object = prevention_prompt_input.get_shortform_feedback()
                    longform_feedback = prevention_prompt_input.get_longform_feedback(prompt_output=prevention_prompt_output)

                    if prevention_pattern == 'mitigation':
                        longform_feedback = prompt_input_object.get_longform_feedback(prompt_output=prompt_output, pattern_to_search_for='Mitigation Explanation',lookahead_assertion='Answer')

                    if prevention_pattern == 'mitigation' or prevention_pattern == 'neither':
                        feedback_for_incorrect_answers += f'''
                        {feedback_header}
                        \n\n\n\n##### Feedback: {shortform_feedback_object.negative_feedback}\n\n\n\n
                        \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n
                        \n\n\n\n##### Recommendation: Please look at the definition of a Prevention {'and Mitigation' if prevention_pattern == 'mitigation' else ''} for assistance.\n\n\n\n'''

                        is_everything_correct = False
                    
                    if prevention_pattern == 'prevention' or prevention_pattern == 'both':
                        feedback_for_correct_answers += f'''
                        {feedback_header}
                        \n\n\n\n##### Feedback: {shortform_feedback_object.positive_feedback}\n\n\n\n
                        \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n'''

        # MITIGATION CHECKS
            feedback_header = f'''\n\n\n## Feedback for Input: Mitigation\n\n\n'''

            mitigation_protective_clothing_prompt_input = RA.get_mitigation_protective_clothing_input()
            mitigation_protective_clothing_prompt_output, mitigation_protective_clothing_pattern = RA.get_prompt_output_and_pattern_matched(mitigation_protective_clothing_prompt_input, LLM)

            shortform_feedback = mitigation_protective_clothing_prompt_input.get_shortform_feedback()
            longform_feedback = mitigation_protective_clothing_prompt_input.get_longform_feedback()

            # Indicating that the mitigation is a protective clothing
            if mitigation_protective_clothing_pattern == True:
                feedback_for_correct_answers += f'''
                {feedback_header}
                \n\n\n\n##### Feedback: {shortform_feedback.positive_feedback}\n\n\n\n
                \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n'''
            
            # Indicating that the mitigation is not a protective clothing
            else:
                mitigation_first_aid_prompt_input = RA.get_mitigation_first_aid_input()
                mitigation_first_aid_prompt_output, mitigation_first_aid_pattern = RA.get_prompt_output_and_pattern_matched(mitigation_first_aid_prompt_input, LLM)

                shortform_feedback = mitigation_first_aid_prompt_input.get_shortform_feedback()
                longform_feedback = mitigation_first_aid_prompt_input.get_longform_feedback()

                # Indicating that the mitigation is an example of first aid
                if mitigation_first_aid_pattern == True:
                        
                    feedback_for_correct_answers += f'''
                    {feedback_header}
                    \n\n\n\n##### Feedback: {shortform_feedback.positive_feedback}\n\n\n\n
                    \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n'''

                # Indicating that the mitigation is neither a protective clothing or an example of first aid
                # This checks whether the inputted mitigation is a prevention or a mitigation 
                else:
                    mitigation_prompt_input = RA.get_mitigation_input()
                    mitigation_prompt_output, mitigation_pattern = RA.get_prompt_output_and_pattern_matched(mitigation_prompt_input, LLM)
                    
                    if mitigation_pattern == 'mitigation' or mitigation_pattern == 'both':
                        feedback_for_correct_answers += f'''
                        {feedback_header}
                        \n\n\n\n##### Feedback: {shortform_feedback.positive_feedback}\n\n\n\n
                        \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n'''

                    if mitigation_pattern == 'prevention':
                        longform_feedback = mitigation_prompt_input.get_longform_feedback(prompt_output=mitigation_prompt_output, pattern_to_search_for='Prevention Explanation',lookahead_assertion='Mitigation')
                    
                    if mitigation_pattern == 'prevention' or mitigation_pattern == 'neither':
                        feedback_for_incorrect_answers += f'''
                        {feedback_header}
                        \n\n\n\n##### Feedback: {shortform_feedback.negative_feedback}\n\n\n\n
                        \n\n\n\n##### Explanation: {longform_feedback}\n\n\n\n
                        \n\n\n\n##### Recommendation: Please look at the definition of a Mitigation {'and Prevention' if mitigation_pattern == 'prevention' else ''} for assistance.\n\n\n\n'''

                        is_everything_correct = False
        
        if feedback_for_incorrect_answers == '# Feedback for Incorrect Answers\n':
            feedback_for_incorrect_answers = '# Congratulations! All your answers are correct!'

        feedback_for_correct_answers += f'''
        \n\n\n## Feedback for Risk Multiplications {field}\n\n\n\n
        \n\n\n\n##### Uncontrolled risk multiplication is: {uncontrolled_risk}\n\n\n\n
        \n\n\n\n##### Controlled risk multiplication is: {controlled_risk}\n\n\n\n'''

        return Result(is_correct=is_everything_correct, feedback=feedback_for_incorrect_answers + '\n\n\n\n\n' + feedback_for_correct_answers)