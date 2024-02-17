import streamlit as st
import pandas as pd
from GoogleSheetsWriter import GoogleSheetsWriter

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

    def get_JSON_output_from_API_call(self, prompt):
        messages = [{"role": "user", "content": prompt}]

        # TODO: Vary max_tokens based on prompt and test different temperatures.
        # NOTE: Lower temperature means more deterministic output.
        LLM_output = openai.ChatCompletion.create(model="gpt-3.5-turbo", 
                                            messages=messages, 
                                            temperature=self.temperature, 
                                            max_tokens=self.max_tokens)
        return LLM_output
    
    def get_model_output(self, prompt):
        LLM_output = self.get_JSON_output_from_API_call(prompt)
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

# Add option in RiskAssessment to specify whether prevention is misclassified as mitigation, 
# is not a suitable prevention, or mitigation is misclassified as prevention, or is not a suitable mitigation

# TODO: Functions can make this code shorter.
 
from typing import Any, TypedDict
import numpy as np

try:
    from .PromptInputs import *
except:
    from PromptInputs import *

try:
    from RiskAssessment import RiskAssessment
    from LLMCaller import LLMWithCandidateLabels, LLMWithGeneratedText
except:
    from .RiskAssessment import RiskAssessment
    from .LLMCaller import LLMWithCandidateLabels, LLMWithGeneratedText, OpenAILLM

class Params(TypedDict):
    pass

class Result(TypedDict):
    is_correct: bool
    feedback: str

def evaluation_function(response: Any, answer: Any, params: Params) -> Result:
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

    if params["is_feedback_text"] == True:
        return Result(is_correct=True, feedback="Thank you for your feedback")
    
    else:
        activity, hazard, how_it_harms, who_it_harms, uncontrolled_likelihood, uncontrolled_severity, uncontrolled_risk, prevention, mitigation, controlled_likelihood, controlled_severity, controlled_risk = np.array(response).flatten()

        RA = RiskAssessment(activity=activity, hazard=hazard, who_it_harms=who_it_harms, how_it_harms=how_it_harms,
                            uncontrolled_likelihood=uncontrolled_likelihood, uncontrolled_severity=uncontrolled_severity,
                            uncontrolled_risk=uncontrolled_risk, prevention=prevention, mitigation=mitigation,
                            controlled_likelihood=controlled_likelihood, controlled_severity=controlled_severity, controlled_risk=controlled_risk,
                            prevention_prompt_expected_output='prevention', mitigation_prompt_expected_output='mitigation',
                            prevention_protected_clothing_expected_output=False,
                            mitigation_protected_clothing_expected_output=False,
                            prevention_first_aid_expected_output=False,
                            mitigation_first_aid_expected_output=False,
                            harm_caused_in_how_it_harms='',
                            hazard_event="",
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

            feedback_for_incorrect_answers = '\n\n\n\n# Feedback for Incorrect Answers\n\n\n\n'
            feedback_for_correct_answers = '\n\n\n\n# Feedback for Correct Answers\n\n\n\n'

            is_everything_correct = True
            
            if is_everything_correct == True:
                first_3_prompt_input_objects = RA.get_list_of_prompt_input_objects_for_first_3_prompts()

                for prompt_input_object in first_3_prompt_input_objects:
                    prompt_output, pattern = RA.get_prompt_output_and_pattern_matched(prompt_input_object, LLM)
                    shortform_feedback = RA.get_shortform_feedback_from_regex_match(prompt_input_object, pattern)

                    field = prompt_input_object.get_field_checked()
                    
                    feedback_header_to_add = f''' 
                    \n\n\n## Feedback for Input: {field}\n\n\n
                    '''

                    feedback_to_add = f'''
                    \n\n\n\n#### Feedback: {shortform_feedback}\n\n\n\n'''
                    
                    if pattern in prompt_input_object.labels_indicating_correct_input:
                        feedback_for_correct_answers += feedback_header_to_add
                        feedback_for_correct_answers += feedback_to_add + '\n\n'
                    
                    else:

                        longform_feedback = prompt_input_object.get_longform_feedback(prompt_output=prompt_output)
                        
                        if longform_feedback != '':
                            feedback_to_add += f'''\n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n'''
                        
                        is_everything_correct = False
                        recommendation = prompt_input_object.get_recommendation()

                        feedback_to_add += f'''\n\n\n\n#### Recommendation: {recommendation}'''

                        feedback_for_incorrect_answers += feedback_header_to_add
                        feedback_for_incorrect_answers += feedback_to_add

                        break
            
            # PREVENTION CHECKS
            if is_everything_correct == True:
                feedback_header = f'''\n\n\n## Feedback for Input: Prevention\n\n\n'''

                prevention_protective_clothing_prompt_input = RA.get_prevention_protective_clothing_input()
                prevention_protective_clothing_prompt_output, prevention_protective_clothing_pattern = RA.get_prompt_output_and_pattern_matched(prevention_protective_clothing_prompt_input, LLM)
                
                # Indicating that the prevention is a protective clothing so is actually a mitigation
                if prevention_protective_clothing_pattern == True:
                    shortform_feedback = prevention_protective_clothing_prompt_input.get_shortform_feedback('negative')
                    longform_feedback = prevention_protective_clothing_prompt_input.get_longform_feedback()
                    recommendation = prevention_protective_clothing_prompt_input.get_recommendation()

                    feedback_for_incorrect_answers += f'''
                    {feedback_header}
                    \n\n\n\n#### Feedback: {shortform_feedback}\n\n\n\n
                    \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n
                    \n\n\n\n#### Recommendation: {recommendation}\n\n\n\n'''

                    is_everything_correct = False
                
                # Indicating that the prevention is not a protective clothing
                else:
                    prevention_first_aid_prompt_input = RA.get_prevention_first_aid_input()
                    prevention_first_aid_prompt_output, prevention_first_aid_pattern = RA.get_prompt_output_and_pattern_matched(prevention_first_aid_prompt_input, LLM)

                    # Indicating that the prevention is an example of first aid so is a mitigation
                    if prevention_first_aid_pattern == True:
                        shortform_feedback = prevention_first_aid_prompt_input.get_shortform_feedback('negative')
                        longform_feedback = prevention_first_aid_prompt_input.get_longform_feedback()
                        recommendation = prevention_first_aid_prompt_input.get_recommendation()

                        feedback_for_incorrect_answers += f'''
                        {feedback_header}
                        \n\n\n\n#### Feedback: {shortform_feedback}\n\n\n\n
                        \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n
                        \n\n\n\n#### Recommendation: {recommendation}\n\n\n\n'''

                        is_everything_correct = False
                    
                    # Indicating that the prevention is neither a protective clothing nor an example of first aid
                    # This checks whether the inputted prevention is a prevention or a mitigation 
                    else:
                        prevention_prompt_input = RA.get_prevention_input()
                        prevention_prompt_output, prevention_pattern = RA.get_prompt_output_and_pattern_matched(prevention_prompt_input, LLM)

                        longform_feedback = prevention_prompt_input.get_longform_feedback(prompt_output=prevention_prompt_output)

                        if prevention_pattern == 'prevention' or prevention_pattern == 'both':
                            feedback_for_correct_answers += f'''
                            {feedback_header}
                            \n\n\n\n#### Feedback: {prevention_prompt_input.get_shortform_feedback('positive')}\n\n\n\n
                            \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n'''

                        if prevention_pattern == 'neither':
                            recommendation = prevention_prompt_input.get_recommendation(recommendation_type='neither')
                            feedback_for_incorrect_answers += f'''
                            {feedback_header}
                            \n\n\n\n#### Feedback: {prevention_prompt_input.get_shortform_feedback('neither')}\n\n\n\n
                            \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n
                            \n\n\n\n#### Recommendation: {recommendation}\n\n\n\n'''

                            is_everything_correct = False

                        if prevention_pattern == 'mitigation':
                            longform_feedback = prevention_prompt_input.get_longform_feedback(prompt_output=prompt_output, pattern_to_search_for='Mitigation Explanation')
                            recommendation = prevention_prompt_input.get_recommendation(recommendation_type='misclassification')

                            feedback_for_incorrect_answers += f'''
                            {feedback_header}
                            \n\n\n\n#### Feedback: {prevention_prompt_input.get_shortform_feedback('misclassification')}\n\n\n\n
                            \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n
                            \n\n\n\n#### Recommendation: {recommendation}\n\n\n\n'''

                            is_everything_correct = False

                # MITIGATION CHECKS
                feedback_header = f'''\n\n\n## Feedback for Input: Mitigation\n\n\n'''

                mitigation_protective_clothing_prompt_input = RA.get_mitigation_protective_clothing_input()
                mitigation_protective_clothing_prompt_output, mitigation_protective_clothing_pattern = RA.get_prompt_output_and_pattern_matched(mitigation_protective_clothing_prompt_input, LLM)

                shortform_feedback = mitigation_protective_clothing_prompt_input.get_shortform_feedback('positive')
                longform_feedback = mitigation_protective_clothing_prompt_input.get_longform_feedback()

                # Indicating that the mitigation is a protective clothing
                if mitigation_protective_clothing_pattern == True:
                    feedback_for_correct_answers += f'''
                    {feedback_header}
                    \n\n\n\n#### Feedback: {shortform_feedback}\n\n\n\n
                    \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n'''
                
                # Indicating that the mitigation is not a protective clothing
                else:
                    mitigation_first_aid_prompt_input = RA.get_mitigation_first_aid_input()
                    mitigation_first_aid_prompt_output, mitigation_first_aid_pattern = RA.get_prompt_output_and_pattern_matched(mitigation_first_aid_prompt_input, LLM)

                    shortform_feedback = mitigation_first_aid_prompt_input.get_shortform_feedback('positive')
                    longform_feedback = mitigation_first_aid_prompt_input.get_longform_feedback()

                    # Indicating that the mitigation is an example of first aid
                    if mitigation_first_aid_pattern == True:
                            
                        feedback_for_correct_answers += f'''
                        {feedback_header}
                        \n\n\n\n#### Feedback: {shortform_feedback}\n\n\n\n
                        \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n'''

                    # Indicating that the mitigation is neither a protective clothing or an example of first aid
                    # This checks whether the inputted mitigation is a prevention or a mitigation 
                    else:
                        mitigation_prompt_input = RA.get_mitigation_input()
                        mitigation_prompt_output, mitigation_pattern = RA.get_prompt_output_and_pattern_matched(mitigation_prompt_input, LLM)

                        longform_feedback = mitigation_prompt_input.get_longform_feedback(prompt_output=mitigation_prompt_output)

                        if mitigation_pattern == 'mitigation' or mitigation_pattern == 'both':
                            feedback_for_correct_answers += f'''
                            {feedback_header}
                            \n\n\n\n#### Feedback: {mitigation_prompt_input.get_shortform_feedback('positive')}\n\n\n\n
                            \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n'''
                        
                        if mitigation_pattern == 'neither':
                            recommendation = mitigation_prompt_input.get_recommendation(recommendation_type='neither')

                            feedback_for_incorrect_answers += f'''
                            {feedback_header}
                            \n\n\n\n#### Feedback: {mitigation_prompt_input.get_shortform_feedback('neither')}\n\n\n\n
                            \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n
                            \n\n\n\n#### Recommendation: {recommendation}\n\n\n\n'''

                            is_everything_correct = False

                        if mitigation_pattern == 'prevention':
                            longform_feedback = mitigation_prompt_input.get_longform_feedback(prompt_output=mitigation_prompt_output, pattern_to_search_for='Prevention Explanation')
                            recommendation = mitigation_prompt_input.get_recommendation(recommendation_type='misclassification')

                            feedback_for_incorrect_answers += f'''
                            {feedback_header}
                            \n\n\n\n#### Feedback: {mitigation_prompt_input.get_shortform_feedback('misclassification')}\n\n\n\n
                            \n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n
                            \n\n\n\n#### Recommendation: {recommendation}\n\n\n\n'''
                            
                            is_everything_correct = False

            if is_everything_correct == True:
                feedback_for_incorrect_answers = '# Congratulations! All your answers are correct!'

            feedback_for_correct_answers += f'''
            \n\n\n## Feedback for Risk Multiplications\n\n\n\n
            \n\n\n\n#### Uncontrolled risk multiplication is: {uncontrolled_risk}\n\n\n\n
            \n\n\n\n#### Controlled risk multiplication is: {controlled_risk}\n\n\n\n'''

            return Result(is_correct=is_everything_correct, feedback=feedback_for_incorrect_answers + '\n\n\n\n\n' + feedback_for_correct_answers)
st.title('Risk Assessment Exercise')
# st.subheader('Learning Objectives')
# 'Before completing this exercise, please read the following learning objectives:'
# with st.expander('See Learning Objectives'):
#     'On completing this Risk Assessment Exercise, students should be able to:'
#     st.markdown('''
#                 1. List the fields necessary to complete a Risk Assessment
#                 2. Identify the difference between a prevention and mitigation measure
#                 3. Understand how the risk score is calculated
#                 4. Understand the difference between an uncontrolled risk and a controlled risk
#                 5. Fill out a Risk Assessment for any activity''')

st.subheader('Overview: This exercise gives feedback on Risk Assessments.')

st.title('Instructions')
st.markdown('''
            1. Think of an activity that involves risk, e.g. cooking, playing sports, driving, cycling, or crossing the road.
            2. In the "Helpful Resources" below, read the "Input field definitions" and the "Example Risk Assessment".
            3. Fill out the "Risk Assessment Fields" exercise.
                - **NOTE: You should only include one example in each field, e.g. for the 'Mitigation' field, only list one mitigation, e.g. "Wear a helmet"**
            4. When you are finished, you can give feedback to me by filling out the feedback form at the bottom. Many Thanks, Charlie''')
# with st.expander('See Additional Instructions'):
#     st.markdown('''
#                 1. Fill out the "Risk Assessment Fields" exercise below. 
#                     - NOTE: You should only include one example in each field, e.g. for the 'Prevention' field, only list one prevention.
#                     - If you're stuck, use the "Helpful Resources" below.
#                 3. Click submit. Your answers will then be processed (which takes around 10 seconds).
#                 4. Afterwards, feedback will be given. 
#                     - If you got something wrong, you should use the feedback provided to improve your answers.
#                     - If you got everything correct, you can click the "See full feedback" dropdown to see why you are correct.
#                 5. When you are finished, you can give feedback to me by filling out the feedback form at the bottom. Many Thanks, Charlie''')
    
st.title('Helpful Resources')

st.subheader('Input Field Definitions')
'Please read the definitions of the input fields below:'
with st.expander('Click to see Input Field Definitions'):
    definitions = {
        'Field': ['Activity', 'Hazard', 'How it harms', 'Who it harms', 'Prevention', 'Mitigation', 'Likelihood', 'Severity', 'Risk', 'Uncontrolled Likelihood', 'Uncontrolled Severity', 'Uncontrolled Risk', 'Controlled Likelihood', 'Controlled Severity', 'Controlled Risk'],
        'Definition': [
            'Activity involving the hazard',
            'Dangerous phenomenon, object, human activity or condition.',
            'One-sentence explanation of how the hazard can cause harm.',
            'Individuals or groups at risk of harm from the hazard.',
            'Action which reduces the likelihood of the hazard causing harm.',
            'Action which reduces the harm caused by the hazard.',
            'The probability that the hazard causes harm. Ranges from 1-5.',
            'The degree of harm that the hazard can cause. Ranges from 1-5.',
            'Calculated using Risk = Likelihood x Severity.',
            'Likelihood before prevention measure applied. Ranges from 1-5.',
            'Severity before mitigation measure applied. Ranges from 1-5.',
            'Risk before prevention/mitigation applied.',
            'Likelihood after prevention measure applied. Ranges from 1-5.',
            'Severity after mitigation measure applied. Ranges from 1-5.',
            'Risk after prevention/mitigation applied.',
        ],
        'Input format': ['Text', 'Text', 'Text', 'Text', 'Text', 'Text', 'Integer', 'Integer', 'Integer', 'Integer', 'Integer', 'Integer', 'Integer', 'Integer', 'Integer']
    }

    # Create DataFrame from the dictionary
    df_markdown = pd.DataFrame(definitions)

    # Display DataFrame without index column in Streamlit
    st.write(df_markdown)

st.subheader('Example Risk Assessment')
'Please look at the example risk assessment below:'
with st.expander('Click to see Example Risk Assessment'):
    example_risk_assessment = {
        'Field': ['Activity', 'Hazard', 'How it harms', 'Who it harms', 'Uncontrolled Likelihood', 
                'Uncontrolled Severity', 'Uncontrolled Risk', 'Prevention', 'Mitigation', 
                'Controlled Likelihood', 'Controlled Severity', 'Controlled Risk'],
        'Example': ['Cycling', 'Reckless drivers', 'Collision with cars causes impact injury', 'Cyclists', 1, 
                    4, 4 , 'Use high viz and lights', 'Wear a helmet', 1, 2, 2]
    }

    df = pd.DataFrame(example_risk_assessment)
    st.table(df)


st.title('Risk Assessment Fields')
with st.form('risk_assessment'):
    # activity = st.text_input('Activity', value='Fluids laboratory')
    # hazard = st.text_input('Hazard', value="Water being spilt on the floor")
    # how_it_harms = st.text_input('How does the hazard cause harm?', value="Injuries caused by possible slipping on wet floor")
    # who_it_harms = st.text_input('Who does the hazard harm?', value="Students")
    # uncontrolled_likelihood = st.text_input('Uncontrolled Likelihood (enter an integer between 1 and 5)', value='2')
    # uncontrolled_severity = st.text_input('Uncontrolled Severity (enter an integer between 1 and 5)', value='2')
    # uncontrolled_risk = st.text_input('Uncontrolled Risk (enter an integer between 1 and 25)', value='4')
    # prevention = st.text_input('Prevention', value="Do not move the water tank when it is full")
    # mitigation=st.text_input('Mitigation', value="""If someone gets injured due to slipping, apply an ice pack to the injured area and seek medical advice without delay.""")
    # controlled_likelihood = st.text_input('Controlled Likelihood (enter an integer between 1 and 5)', value='1')
    # controlled_severity = st.text_input('Controlled Severity (enter an integer between 1 and 5)', value='1')
    # controlled_risk = st.text_input('Controlled Risk (enter an integer between 1 and 25)', value='1')
    # submit_button = st.form_submit_button("Submit")

    activity = st.text_input('Activity', value='Mucking out horse\'s stable')
    hazard = st.text_input('Hazard', value="Horse")
    how_it_harms = st.text_input('Harm caused by this event', value="Impact injuries")
    who_it_harms = st.text_input('Who is harmed by this event', value="Stable hand")
    uncontrolled_likelihood = st.text_input('Uncontrolled Likelihood (enter an integer between 1 and 5)', value='2')
    uncontrolled_severity = st.text_input('Uncontrolled Severity (enter an integer between 1 and 5)', value='2')
    uncontrolled_risk = st.text_input('Uncontrolled Risk (enter an integer between 1 and 25)', value='4')
    prevention = st.text_input('Prevention', value="Do not stand behind the horse")
    mitigation=st.text_input('Mitigation', value="""Wear a helmet and body protector when mucking out the horse's stable""")
    controlled_likelihood = st.text_input('Controlled Likelihood (enter an integer between 1 and 5)', value='1')
    controlled_severity = st.text_input('Controlled Severity (enter an integer between 1 and 5)', value='1')
    controlled_risk = st.text_input('Controlled Risk (enter an integer between 1 and 25)', value='1')
    submit_button = st.form_submit_button("Submit")
                      
    # activity = st.text_input('Activity', value='Fluids laboratory')
    # hazard = st.text_input('Hazard', value="Ink spillage")
    # how_it_harms = st.text_input('How does the hazard cause harm?', value="Serious eye damage")
    # who_it_harms = st.text_input('Who does the hazard harm?', value="Students")
    # uncontrolled_likelihood = st.text_input('Uncontrolled Likelihood (enter an integer between 1 and 5)', value='2')
    # uncontrolled_severity = st.text_input('Uncontrolled Severity (enter an integer between 1 and 5)', value='2')
    # uncontrolled_risk = st.text_input('Uncontrolled Risk (enter an integer between 1 and 25)', value='4')
    # prevention = st.text_input('Prevention', value="Wear safety glasses")
    # mitigation = st.text_input('Mitigation', value="Wash eyes with water")
    # controlled_likelihood = st.text_input('Controlled Likelihood (enter an integer between 1 and 5)', value='1')
    # controlled_severity = st.text_input('Controlled Severity (enter an integer between 1 and 5)', value='1')
    # controlled_risk = st.text_input('Controlled Risk (enter an integer between 1 and 25)', value='1')
    # submit_button = st.form_submit_button("Submit")

    # activity = st.text_input('Activity')
    # hazard = st.text_input('Hazard')
    # how_it_harms = st.text_input('How does the hazard cause harm?')
    # who_it_harms = st.text_input('Who does the hazard harm?')
    # uncontrolled_likelihood = st.text_input('Uncontrolled Likelihood (enter an integer between 1 and 5)') 
    # uncontrolled_severity = st.text_input('Uncontrolled Severity (enter an integer between 1 and 5)')
    # uncontrolled_risk = st.text_input('Uncontrolled Risk (enter an integer between 1 and 25)') 
    # prevention = st.text_input('Prevention')
    # mitigation = st.text_input('Mitigation')
    # controlled_likelihood = st.text_input('Controlled Likelihood (enter an integer between 1 and 5)') 
    # controlled_severity = st.text_input('Controlled Severity (enter an integer between 1 and 5)')
    # controlled_risk = st.text_input('Controlled Risk (enter an integer between 1 and 25)')
    # submit_button = st.form_submit_button("Submit")

    # TODO: Add try except for when they are not connected to the internet.
    if submit_button:
        with st.spinner('Getting Risk Assessment Feedback...'):
            response = [activity, hazard, how_it_harms, who_it_harms, uncontrolled_likelihood, 
                        uncontrolled_severity, uncontrolled_risk, prevention, mitigation, 
                        controlled_likelihood, controlled_severity, controlled_risk]
            params: Params = {"is_feedback_text": False}

            result = evaluation_function(response=response, answer='', params=params)
            is_correct = result['is_correct']
            feedback = result['feedback']

            st.write(feedback)
                
            if 'feedback' not in st.session_state:
                st.session_state.feedback = [feedback]
                # st.session_state.feedback = full_feedback
            else:
                st.session_state.feedback.append(feedback)

st.title('Your feedback for me')
with st.expander('Please fill out this form so I can improve the Exercise!'):
    with st.form('feedback_from_user'):
        options = ['Yes', 'No']
        option_likert = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
        name = st.text_input('1/7) First Name')
        is_feedback_accurate = st.radio('2/7) Is the feedback you received accurate?', options=options)
        why_not_accurate = st.text_input('If you answered No, why is the feedback accurate?')
        is_feedback_helpful = st.radio('3/7) Is the feedback you received helpful?', options=option_likert)
        is_feedback_specific  = st.radio('4/7) Is the feedback you received specific, i.e. tailored specifically to the answers you gave?', options=options)
        why_not_specific = st.text_input('If you answered No, why is the feedback not specific?')
        is_feedback_constructive = st.radio('5/7) Is the feedback you received constructive, i.e. helped you understand why an answer was wrong and made helpful suggestions on how to improve your answer.', options=options)
        why_not_constructive = st.text_input('If you answered No, why is the feedback not constructive?')
        are_instructions_clear = st.radio('6/7) Do you think the instructions given for this exercise were clear?', options=options)
        why_not_clear = st.text_input('If you answered No, why are the instructions not clear?')
        # st.subheader('Learning Objectives:')
        # 'On completing this Risk Assessment Exercise, students should be able to:'
        # st.markdown('''
        #             1. List the fields necessary to complete a Risk Assessment
        #             2. Identify the difference between a prevention and mitigation measure
        #             3. Understand how the risk score is calculated
        #             4. Understand the difference between an uncontrolled risk and a controlled risk
        #             5. Fill out a Risk Assessment for any activity''')
        # ''
        # ''
        # learning_outcomes = st.radio('6/7) Do you feel you have obtained all the learning outcomes specified for this exercise?', options=options)
        # why_not_learning_outcomes = st.text_input('If you answered No, why do you think you have not obtained all the learning outcomes?')
        general_feedback = st.text_input('7/7) Any other general feedback?')
        submit_button = st.form_submit_button("Submit")
        google_sheets_writer = GoogleSheetsWriter(sheet_name='Sheet1', spreadsheet_id='1O6ztnca_NPC0TXxmSTrw5xV51vUow0I8nAebEsEntbQ')

        RiskAssessment_string = f'''RiskAssessment(
            activity="{activity}",
            hazard="{hazard}",
            who_it_harms="{who_it_harms}",
            how_it_harms="{how_it_harms}",
            uncontrolled_likelihood="{uncontrolled_likelihood}",
            uncontrolled_severity="{uncontrolled_severity}",
            uncontrolled_risk="{uncontrolled_risk}",
            prevention="{prevention}",
            mitigation="{mitigation}",
            controlled_likelihood="{controlled_likelihood}",
            controlled_severity="{controlled_severity}",
            controlled_risk="{controlled_risk}",
            prevention_prompt_expected_output="prevention",
            mitigation_prompt_expected_output="mitigation"
        )'''

        if submit_button:
            if 'feedback' in st.session_state:
                google_sheets_writer.write_to_sheets(new_line_data=[
                    name,
                    RiskAssessment_string,
                    '\n\n\n'.join(st.session_state.feedback),
                    is_feedback_accurate,
                    why_not_accurate,
                    is_feedback_helpful,
                    is_feedback_specific,
                    why_not_specific,
                    is_feedback_constructive,
                    why_not_constructive,
                    are_instructions_clear,
                    why_not_clear,
                    general_feedback
                    ])
                st.write('Thank you very much for your feedback! It will be taken on board!')
            else:
                st.write('Please submit a risk assessment first')