

# Low hanging fruit:
# TODO: Improve few shot prompting examples so they don't parrot back input prompt
# TODO: Try using chain of thought prompt engineering for the mitigation prompt
# TODO: Try using Llama inference endpoint
# TODO: Try using Llama inference API but specify the number of tokens you want to receive
# TODO: Update question description in lambda feedback making it clear that 
# only one mitigation, one prevention and one 'how it harms' is specified

# Add option in RiskAssessment to specify whether prevention is misclassified as mitigation, 
# is not a suitable prevention, or mitigation is misclassified as prevention, or is not a suitable mitigation

from GoogleSheetsWriter import GoogleSheetsWriter
import numpy as np
from typing import Type, Any, TypedDict
import streamlit as st
import pandas as pd

try:
    from .PromptInputs import *
    from .RiskAssessment import RiskAssessment
    from .LLMCaller import *
    
    
except ImportError:
    from PromptInputs import *
    from RiskAssessment import RiskAssessment
    from LLMCaller import *

class Result(TypedDict):
    is_correct: bool
    feedback: str

# Add option in RiskAssessment to specify whether prevention is misclassified as mitigation, 
# is not a suitable prevention, or mitigation is misclassified as prevention, or is not a suitable mitigation

# TODO: Functions can make this code shorter.

class Params(TypedDict):
    is_feedback_text: bool
    is_risk_matrix: bool
    is_risk_assessment: bool

def provide_feedback_on_risk_matrix(response):
        risk_matrix = np.array(response)

        risk_matrix_flattened = np.array(response).flatten()
        for value in risk_matrix_flattened:
            if value == '':
                return Result(is_correct=False, feedback="Please fill in all the fields")
            else:
                try:
                    int_value = int(value)
                except ValueError:
                    return Result(is_correct=False, feedback="Please enter an integer for all fields.")

        risk_matrix_dict = {'uncontrolled likelihood': int(risk_matrix[0, 0]),
                    'uncontrolled severity': int(risk_matrix[0, 1]),
                    'controlled likelihood': int(risk_matrix[1, 0]), 
                    'controlled severity': int(risk_matrix[1, 1])}

        is_correct = True
        feedback = f''''''

        for key in risk_matrix_dict.keys():
            if risk_matrix_dict[key] > 4 or risk_matrix_dict[key] < 1:
                is_correct = False
                feedback += f'''\n\n\n\n\n##### The {key} is incorrect. As per the likelihood and severity conventions above, the likelihood and severity should be between 1 and 4.\n\n\n\n'''

        uncontrolled_likelihood = risk_matrix_dict['uncontrolled likelihood']
        uncontrolled_severity = risk_matrix_dict['uncontrolled severity']
        uncontrolled_risk = int(risk_matrix[0, 2])
        controlled_likelihood = risk_matrix_dict['controlled likelihood']
        controlled_severity = risk_matrix_dict['controlled severity']
        controlled_risk = int(risk_matrix[1, 2])

        feedback += '\n'

        # Comparing Uncontrolled and Controlled Rows
                
        # Likelihoods
        if is_correct == True: # Only write feedback if all previous checks have passed
            if uncontrolled_likelihood <= controlled_likelihood:
                feedback += f'''\n\n\n\n\n##### Likelihood values are incorrect. Since an effective prevention measure has been implemented ("taking care when cross the road"), the controlled likelihood should be less than the uncontrolled likelihood.\n\n\n\n'''
                is_correct = False

        # Severities
        if is_correct == True: # Only write feedback if all previous checks have passed
            if uncontrolled_severity != controlled_severity:
                feedback += f'''\n\n\n\n\n##### Severity values are incorrect. The uncontrolled and controlled severity should be the same since no mitigation measure has been implemented.\n\n\n\n''' 
                is_correct = False

        # --- Checking Uncontrolled Row ---
        
        # Likelihoods
        if is_correct == True: # Only write feedback if all previous checks have passed
            if uncontrolled_likelihood != 4:
                feedback += f'''\n\n\n\n\n##### An uncontrolled likelihood of {uncontrolled_likelihood} is incorrect. The convention is that all uncontrolled risks have a likelihood of 4. If you didn't look or listen when crossing the road, you would almost certainly be harmed.\n\n\n\n'''
                is_correct = False

        # Severities
        if is_correct == True:
            if uncontrolled_severity == 1:
                feedback += f'''\n\n\n\n\n##### An uncontrolled severity of 1 is incorrect. As by the above severity convention, a severity of 1 indicates that a car crashing into a pedestrian causes "minor injury or property damage". The harm will be greater than this.\n\n\n\n''' 
                is_correct = False

        # Multiplications
        if is_correct == True: # Only write feedback if all previous checks have passed
            if uncontrolled_likelihood * uncontrolled_severity != uncontrolled_risk:
                feedback += f'''\n\n\n\n\n##### Uncontrolled risk multiplication is incorrect. Make sure the risk is the likelihood multiplied by the severity.\n\n\n\n'''
                is_correct = False

        # --- Checking Controlled Row ---
                
        # Likelihoods
        if is_correct == True: # Only write feedback if all previous checks have passed
            if controlled_likelihood == 1:
                feedback += f'''\n\n\n\n\n##### A controlled likelihood of 1 is incorrect. A controlled likelihood of 1 indicates that the control measure is implemented passively whereas you have to activily pay attention when cross the road.\n\n\n\n'''
                is_correct = False

            if controlled_likelihood == 2:
                feedback += f'''\n\n\n\n\n##### Correct controlled likelihood. A controlled likelihood of 2 indicates that the control measure of "taking care when crossing the road" is implemented actively.\n\n\n\n''' 
            
            if controlled_likelihood == 3:
                feedback += f'''\n\n\n\n\n##### A controlled likelihood of 3 is incorrect. A controlled likelihood of 3 indicates that the control measure is not effective and the likelihood is "possible".\n\n\n\n''' 
                is_correct = False
            
            if controlled_likelihood == 4:
                feedback += f'''\n\n\n\n\n##### A controlled likelihood of 4 is incorrect. A controlled likelihood of 4 indicates that the control measure is effective and the likelihood is "likely".\n\n\n\n''' 
                is_correct = False

        # Severities
        if is_correct == True: # Only write feedback if all previous checks have passed
            if controlled_severity == 1:
                feedback += f'''\n\n\n\n\n##### A controlled severity of 1 is incorrect. As by the above severity convention, a severity of 1 indicates that a car crashing into a pedestrian causes "minor injury or property damage". The harm will be greater than this.\n\n\n\n''' 
                is_correct = False

        # Multiplications
        if is_correct == True: # Only write feedback if all previous checks have passed
            if controlled_likelihood * controlled_severity != controlled_risk:
                feedback += f'''\n\n\n\n\n##### Controlled risk multiplication is incorrect. Make sure the risk is the likelihood multiplied by the severity.\n\n\n\n'''
                is_correct = False

        return Result(is_correct=is_correct, feedback=feedback)

def provide_feedback_on_control_measure_input(control_measure_input_field: str,
                                              control_measure_prompt_input: Type[PromptInput],
                                              control_measure_prompt_output: str,
                                              control_measure_prompt_pattern: str,
                                              feedback_for_correct_answers: str,
                                              feedback_for_incorrect_answers: str,
                                              is_everything_correct: bool,
                                              risk_assessment: RiskAssessment,
                                              LLM: Type[LLMCaller]):
    
    prevention_prompt_feedback = control_measure_prompt_input.get_longform_feedback(prompt_output=control_measure_prompt_output, start_string='Prevention Explanation', end_string='Mitigation Explanation')
    mitigation_prompt_feedback = control_measure_prompt_input.get_longform_feedback(prompt_output=control_measure_prompt_output, start_string='Mitigation Explanation', end_string='Answer')

    if control_measure_input_field == 'prevention':
        other_control_measure_input_field = 'mitigation'
        control_measure_prompt_feedback = prevention_prompt_feedback
        other_control_measure_prompt_feedback = mitigation_prompt_feedback

    if control_measure_input_field == 'mitigation':
        other_control_measure_input_field = 'prevention'
        control_measure_prompt_feedback = mitigation_prompt_feedback
        other_control_measure_prompt_feedback = prevention_prompt_feedback
    
    feedback_header = f'\n\n\n\n\n## Feedback for Input: {control_measure_input_field.capitalize()}\n\n\n\n'
    
    if control_measure_prompt_pattern == 'both':
        prompt_input_for_summarizing_control_measure_prompt_feedback = risk_assessment.get_feedback_summary_input()
        summary_of_control_measure_prompt_feedback, _ = risk_assessment.get_prompt_output_and_pattern_matched(prompt_input_object=prompt_input_for_summarizing_control_measure_prompt_feedback, LLM_caller=LLM, control_measure_type=control_measure_input_field, feedback=control_measure_prompt_feedback)
        summary_of_other_control_measure_prompt_feedback, _ = risk_assessment.get_prompt_output_and_pattern_matched(prompt_input_object=prompt_input_for_summarizing_control_measure_prompt_feedback, LLM_caller=LLM, control_measure_type=other_control_measure_input_field, feedback=other_control_measure_prompt_feedback)

        feedback_for_correct_answers += f'''
        {feedback_header}
        \n\n\n\n#### Feedback: {control_measure_prompt_input.get_shortform_feedback('both')}\n\n\n\n
        #### {control_measure_input_field.capitalize()} Explanation: {summary_of_control_measure_prompt_feedback}\n\n\n\n
        #### {other_control_measure_input_field.capitalize()} Explanation: {summary_of_other_control_measure_prompt_feedback}\n\n\n\n'''

    if control_measure_prompt_pattern == control_measure_input_field:
        prompt_input_for_summarizing_control_measure_prompt_feedback = risk_assessment.get_feedback_summary_input()
        summary_of_control_measure_prompt_feedback, _ = risk_assessment.get_prompt_output_and_pattern_matched(prompt_input_object=prompt_input_for_summarizing_control_measure_prompt_feedback, LLM_caller=LLM, control_measure_type=control_measure_input_field, feedback=control_measure_prompt_feedback)

        feedback_for_correct_answers += f'''
        {feedback_header}
        \n\n\n\n#### Explanation: {summary_of_control_measure_prompt_feedback}\n\n\n\n'''

    if control_measure_prompt_pattern == 'neither':
        prompt_input_for_summarizing_control_measure_prompt_feedback = risk_assessment.get_feedback_summary_input()
        summary_of_control_measure_prompt_feedback, _ = risk_assessment.get_prompt_output_and_pattern_matched(prompt_input_object=prompt_input_for_summarizing_control_measure_prompt_feedback, LLM_caller=LLM, control_measure_type=control_measure_input_field, feedback=control_measure_prompt_feedback)

        recommendation = control_measure_prompt_input.get_recommendation(recommendation_type='neither')
        feedback_for_incorrect_answers += f'''
        {feedback_header}
        \n\n\n\n#### Explanation: {summary_of_control_measure_prompt_feedback}\n\n\n\n
        \n\n\n\n#### Recommendation: {recommendation}\n\n\n\n'''

        is_everything_correct = False

    if control_measure_prompt_pattern == other_control_measure_input_field:
        prompt_input_for_summarizing_control_measure_prompt_feedback = risk_assessment.get_feedback_summary_input()
        summary_of_other_control_measure_prompt_feedback, _ = risk_assessment.get_prompt_output_and_pattern_matched(prompt_input_object=prompt_input_for_summarizing_control_measure_prompt_feedback, LLM_caller=LLM, control_measure_type=other_control_measure_input_field, feedback=other_control_measure_prompt_feedback)

        recommendation = control_measure_prompt_input.get_recommendation(recommendation_type='misclassification')
        feedback_for_incorrect_answers += f'''
        {feedback_header}
        \n\n\n\n#### Feedback: {control_measure_prompt_input.get_shortform_feedback('misclassification')}\n\n\n\n
        \n\n\n\n#### Explanation: {summary_of_other_control_measure_prompt_feedback}\n\n\n\n
        \n\n\n\n#### Recommendation: {recommendation}\n\n\n\n'''

        is_everything_correct = False

    return feedback_for_correct_answers, feedback_for_incorrect_answers, is_everything_correct

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
    
    if params["is_risk_matrix"] == True:
        return provide_feedback_on_risk_matrix(response)
    
    if params["is_risk_assessment"] == True:
        activity, hazard, how_it_harms, who_it_harms, uncontrolled_likelihood, uncontrolled_severity, uncontrolled_risk, prevention, mitigation, controlled_likelihood, controlled_severity, controlled_risk = np.array(response).flatten()

        # TODO: Do we need a risk domain?
        RA = RiskAssessment(activity=activity, hazard=hazard, who_it_harms=who_it_harms, how_it_harms=how_it_harms,
                            uncontrolled_likelihood=uncontrolled_likelihood, uncontrolled_severity=uncontrolled_severity,
                            uncontrolled_risk=uncontrolled_risk, prevention=prevention, mitigation=mitigation,
                            controlled_likelihood=controlled_likelihood, controlled_severity=controlled_severity, controlled_risk=controlled_risk,
                            prevention_prompt_expected_class='prevention', mitigation_prompt_expected_class='mitigation', risk_domain='')

        input_check_feedback_message = RA.get_input_check_feedback_message()

        if input_check_feedback_message != True:
            return Result(is_correct=False,
                        feedback=f'''\n\n\n\n\n # Feedback:\n\n\n\n\n
                                    \n\n\n\n\n## {input_check_feedback_message}\n\n\n\n\n''')
        
        likelihood_severity_risk_feedback_message = RA.get_likelihood_severity_risk_feedback_message()

        if likelihood_severity_risk_feedback_message != True:
            return Result(is_correct=False,
                        feedback=f'''\n\n\n\n\n # Feedback:\n\n\n\n\n
                                    \n\n\n\n\n## {likelihood_severity_risk_feedback_message}\n\n\n\n\n''')
        
        # LLM = ClaudeSonnetLLM(system_message='', temperature=0.1, max_tokens=200)
        LLM = OpenAILLM(temperature=0.1, max_tokens=400)

        feedback_for_incorrect_answers = '\n\n\n\n# Feedback for Incorrect Answers\n\n\n\n'
        feedback_for_correct_answers = '\n\n\n\n# Feedback for Correct Answers\n\n\n\n'

        fields_for_which_no_information_provided = []

        is_everything_correct = True

        how_it_harms_in_context_prompt_input = RA.get_how_it_harms_in_context_input()
        who_it_harms_in_context_prompt_input = RA.get_who_it_harms_in_context_input()

        first_2_prompt_input_objects = [how_it_harms_in_context_prompt_input, who_it_harms_in_context_prompt_input]
        
        for prompt_input_object in first_2_prompt_input_objects:
            if is_everything_correct == True:
                prompt_output, pattern = RA.get_prompt_output_and_pattern_matched(prompt_input_object, LLM)
                shortform_feedback = RA.get_shortform_feedback_from_regex_match(prompt_input_object, pattern)

                field = prompt_input_object.get_field_checked()
                
                feedback_header_to_add = f''' 
                \n\n\n## Feedback for Input: {field}\n\n\n
                '''

                if pattern not in prompt_input_object.labels_indicating_correct_input:
                    feedback_to_add = f'''
                    \n\n\n\n#### Feedback: {shortform_feedback}\n\n\n\n'''

                    longform_feedback = prompt_input_object.get_longform_feedback(prompt_output=prompt_output)
                    
                    if longform_feedback != '':
                        feedback_to_add += f'''\n\n\n\n#### Explanation: {longform_feedback}\n\n\n\n'''
                    
                    is_everything_correct = False
                    recommendation = prompt_input_object.get_recommendation()

                    feedback_to_add += f'''\n\n\n\n#### Recommendation: {recommendation}'''

                    feedback_for_incorrect_answers += feedback_header_to_add
                    feedback_for_incorrect_answers += feedback_to_add

                    return Result(is_correct=is_everything_correct, feedback=feedback_for_incorrect_answers)

        # PREVENTION CHECKS
        no_information_provided_for_prevention_prompt_input = RA.get_no_information_provided_for_prevention_input()
        no_information_provided_for_prevention_prompt_output, no_information_provided_for_prevention_pattern = RA.get_prompt_output_and_pattern_matched(no_information_provided_for_prevention_prompt_input, LLM)

        if no_information_provided_for_prevention_pattern == 'no information provided':
            fields_for_which_no_information_provided.append('Prevention')
        
        else:
            # TODO: Avoid duplication of the following code:
            LLM = OpenAILLM(temperature=0.1, max_tokens=400)

            harm_caused_and_hazard_event_prompt_input = RA.get_harm_caused_and_hazard_event_input()
            harm_caused_and_hazard_event_prompt_output, harm_caused_and_hazard_event_pattern = RA.get_prompt_output_and_pattern_matched(harm_caused_and_hazard_event_prompt_input, LLM)

            hazard_event = harm_caused_and_hazard_event_pattern.hazard_event
            harm_caused = harm_caused_and_hazard_event_pattern.harm_caused

            # LLM = MistralLLM(model='open-mixtral-8x7b', temperature=0.1, max_tokens=300)
            LLM = ClaudeSonnetLLM(system_message='', temperature=0.1, max_tokens=400)
            control_measure_prompt_with_prevention_input = RA.get_control_measure_prompt_with_prevention_input()
            control_measure_prompt_with_prevention_output, control_measure_prompt_with_prevention_pattern = RA.get_prompt_output_and_pattern_matched(control_measure_prompt_with_prevention_input, LLM, harm_caused=harm_caused, hazard_event=hazard_event)

            feedback_for_correct_answers, feedback_for_incorrect_answers, is_everything_correct = provide_feedback_on_control_measure_input(
                control_measure_input_field='prevention',
                control_measure_prompt_input=control_measure_prompt_with_prevention_input,
                control_measure_prompt_output=control_measure_prompt_with_prevention_output,
                control_measure_prompt_pattern=control_measure_prompt_with_prevention_pattern,
                feedback_for_correct_answers=feedback_for_correct_answers,
                feedback_for_incorrect_answers=feedback_for_incorrect_answers,
                is_everything_correct=is_everything_correct,
                risk_assessment=RA,
                LLM=LLM
            )

        # MITIGATION CHECKS
        no_information_provided_for_mitigation_prompt_input = RA.get_no_information_provided_for_mitigation_input()
        no_information_provided_for_mitigation_prompt_output, no_information_provided_for_mitigation_pattern = RA.get_prompt_output_and_pattern_matched(no_information_provided_for_mitigation_prompt_input, LLM)

        if no_information_provided_for_mitigation_pattern == 'no information provided':
            fields_for_which_no_information_provided.append('Mitigation')
        else:
            # If harm_caused and hazard_event have not already been extracted.
            if no_information_provided_for_prevention_pattern == 'no information provided':
                # LLM = OpenAILLM(temperature=0.1, max_tokens=400)

                harm_caused_and_hazard_event_prompt_input = RA.get_harm_caused_and_hazard_event_input()
                harm_caused_and_hazard_event_prompt_output, harm_caused_and_hazard_event_pattern = RA.get_prompt_output_and_pattern_matched(harm_caused_and_hazard_event_prompt_input, LLM)

                hazard_event = harm_caused_and_hazard_event_pattern.hazard_event
                harm_caused = harm_caused_and_hazard_event_pattern.harm_caused
            
            # LLM = MistralLLM(model='open-mixtral-8x7b', temperature=0.1, max_tokens=300)
            LLM = ClaudeSonnetLLM(system_message='', temperature=0.1, max_tokens=400)
            
            control_measure_prompt_with_mitigation_input = RA.get_control_measure_prompt_with_mitigation_input()
            control_measure_prompt_with_mitigation_output, control_measure_prompt_with_mitigation_pattern = RA.get_prompt_output_and_pattern_matched(control_measure_prompt_with_mitigation_input, LLM, harm_caused=harm_caused, hazard_event=hazard_event)
            
            feedback_for_correct_answers, feedback_for_incorrect_answers, is_everything_correct = provide_feedback_on_control_measure_input(
                control_measure_input_field='mitigation',
                control_measure_prompt_input=control_measure_prompt_with_mitigation_input,
                control_measure_prompt_output=control_measure_prompt_with_mitigation_output,
                control_measure_prompt_pattern=control_measure_prompt_with_mitigation_pattern,
                feedback_for_correct_answers=feedback_for_correct_answers,
                feedback_for_incorrect_answers=feedback_for_incorrect_answers,
                is_everything_correct=is_everything_correct,
                risk_assessment=RA,
                LLM=LLM
            )

        if is_everything_correct == True:
            feedback_for_incorrect_answers = '# Congratulations! All your answers are correct!'
        
        if fields_for_which_no_information_provided == []:
            no_information_provided_message = ''
        else:
            no_information_provided_message = f'\n\n\n\n\n## Fields for which no information is provided and hence no feedback given: {", ".join(fields_for_which_no_information_provided)}\n\n\n\n\n'

        if fields_for_which_no_information_provided != ['Prevention', 'Mitigation']:
            hazard_event_and_harm_caused_inferred_message = f'''## The following were inferred from your answers: \n\n\n\n\n
            \n\n\n\n\n### Event that leads to harm: "{hazard_event}"\n\n\n\n\n
            \n\n\n\n\n### Harm caused to '{RA.who_it_harms}': "{harm_caused}".\n\n\n\n
            \n\n\n\n\n### If they are incorrect, please make these more explicit in the "Hazard" and "How it harms" fields.\n\n\n\n\n'''
        else:
            hazard_event_and_harm_caused_inferred_message = ''
        
        feedback_for_correct_answers += f'''
        \n\n\n\n### There are no errors in your likelihood, severity, and risk values.\n\n\n\n'''

        return Result(is_correct=is_everything_correct, feedback=hazard_event_and_harm_caused_inferred_message + '\n\n\n\n\n' + feedback_for_incorrect_answers + '\n\n\n\n\n' + feedback_for_correct_answers + '\n\n\n\n\n' + no_information_provided_message)
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
repo_link = "https://github.com/lambda-feedback/riskAssessment"
st.markdown(f'You can find the code repository for this exercise [here]({repo_link}).')

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
    activity = st.text_input('Activity', value='Fluids laboratory')
    hazard = st.text_input('Hazard', value="Water being spilt on the floor")
    how_it_harms = st.text_input('How does the hazard cause harm?', value="Injuries caused by possible slipping on wet floor")
    who_it_harms = st.text_input('Who does the hazard harm?', value="Students")
    uncontrolled_likelihood = st.text_input('Uncontrolled Likelihood (enter an integer between 1 and 5)', value='2')
    uncontrolled_severity = st.text_input('Uncontrolled Severity (enter an integer between 1 and 5)', value='2')
    uncontrolled_risk = st.text_input('Uncontrolled Risk (enter an integer between 1 and 25)', value='4')
    prevention = st.text_input('Prevention', value="Do not move the water tank when it is full")
    mitigation=st.text_input('Mitigation', value="""If someone gets injured due to slipping, apply an ice pack to the injured area and seek medical advice without delay.""")
    controlled_likelihood = st.text_input('Controlled Likelihood (enter an integer between 1 and 5)', value='1')
    controlled_severity = st.text_input('Controlled Severity (enter an integer between 1 and 5)', value='1')
    controlled_risk = st.text_input('Controlled Risk (enter an integer between 1 and 25)', value='1')
    submit_button = st.form_submit_button("Submit")

    # activity = st.text_input('Activity', value='Mucking out horse\'s stable')
    # hazard = st.text_input('Hazard', value="Horse")
    # how_it_harms = st.text_input('Harm caused by this event', value="Impact injuries")
    # who_it_harms = st.text_input('Who is harmed by this event', value="Stable hand")
    # uncontrolled_likelihood = st.text_input('Uncontrolled Likelihood (enter an integer between 1 and 5)', value='2')
    # uncontrolled_severity = st.text_input('Uncontrolled Severity (enter an integer between 1 and 5)', value='2')
    # uncontrolled_risk = st.text_input('Uncontrolled Risk (enter an integer between 1 and 25)', value='4')
    # prevention = st.text_input('Prevention', value="Do not stand behind the horse")
    # mitigation=st.text_input('Mitigation', value="""Wear a helmet and body protector when mucking out the horse's stable""")
    # controlled_likelihood = st.text_input('Controlled Likelihood (enter an integer between 1 and 5)', value='1')
    # controlled_severity = st.text_input('Controlled Severity (enter an integer between 1 and 5)', value='1')
    # controlled_risk = st.text_input('Controlled Risk (enter an integer between 1 and 25)', value='1')
    # submit_button = st.form_submit_button("Submit")
                      
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
            params: Params = {"is_feedback_text": False, "is_risk_matrix": False, "is_risk_assessment": True}

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