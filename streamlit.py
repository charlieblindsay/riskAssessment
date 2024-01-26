import streamlit as st
from evaluation import evaluation_function
import pandas as pd
from GoogleSheetsWriter import GoogleSheetsWriter

st.title('Risk Assessment Exercise')
st.subheader('Learning Objectives')
'Before completing this exercise, please read the learning objectives'
with st.expander('See Learning Objectives'):
    'On completing this Risk Assessment Exercise, students should be able to:'
    st.markdown('''
                1. List the fields necessary to complete a Risk Assessment
                2. Identify the difference between a prevention and mitigation measure
                3. Understand how the risk score is calculated
                4. Understand the difference between an uncontrolled risk and a controlled risk
                5. Fill out a Risk Assessment for any activity''')

st.subheader('Additional Instructions')
'Now use the instructions below to fill out the Risk Assessment fields:'
st.markdown('''
            - You should only include one example in each field, e.g. for the 'Prevention' field, only list one prevention.
            - Your answers will then be processed (which takes around 10 seconds).
            - Afterwards, feedback will be given. You should use the feedback to improve your answers.''' )

st.subheader('To help you, you can use the following dropdowns:')
with st.expander('See what the fields mean'):
    definitions = {
        'Field': ['Prevention', 'Mitigation', 'Likelihood', 'Severity', 'Risk', 'Uncontrolled Risk', 'Controlled Risk'],
        'Definition': [
            'Action which reduces the likelihood of the hazard exists.',
            'Action which reduces the harm caused by the hazard.',
            'The probability that the hazard causes harm. Ranges from 1-5.',
            'The degree of harm that the hazard can cause. Ranges from 1-5.',
            'Calculated using Risk = Likelihood x Severity.',
            'Risk before prevention & mitigation have been applied.',
            'Risk after prevention & mitigation have been applied.'
        ]
    }

    # Create DataFrame from the dictionary
    df_markdown = pd.DataFrame(definitions)

    # Display DataFrame without index column in Streamlit
    st.write(df_markdown)

with st.expander('Example Risk Assessment'):
    example_risk_assessment = {
        'Field': ['Activity', 'Hazard', 'How it harms', 'Who it harms', 'Uncontrolled Likelihood', 
                'Uncontrolled Severity', 'Uncontrolled Risk', 'Prevention', 'Mitigation', 
                'Controlled Likelihood', 'Controlled Severity', 'Controlled Risk'],
        'Example': ['Fluids laboratory', 'Ink spillage', 'Serious eye damage', 'Students', 2, 
                    2, 4, 'Wear safety glasses', 'Wash eyes with water', 1, 1, 1]
    }

    df = pd.DataFrame(example_risk_assessment)
    st.table(df)

with st.form('risk_assessment'):
    activity = st.text_input('Activity', value='Fluids laboratory')
    hazard = st.text_input('Hazard', value="Ink spillage")
    how_it_harms = st.text_input('How does the hazard cause harm?', value="Serious eye damage")
    who_it_harms = st.text_input('Who does the hazard harm?', value="Students")
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

            question_titles = result['question_titles']
            questions = result['question']
            prompt_input_objects = result['prompt_input_objects']
            prompts = result['prompts']
            prompt_outputs = result['prompt_outputs']
            regex_matches = result['regex_matches']
            shortform_feedbacks = result['shortform_feedbacks']
            is_everything_correct = result['is_everything_correct']
            booleans_indicating_which_prompts_need_feedback = result['booleans_indicating_which_prompts_need_feedback']
            controlled_risk = result['controlled_risk']
            uncontrolled_risk = result['uncontrolled_risk']

            n_prompts = len(prompts)

            st.title('Feedback')
            feedback = f'''
                ------ FEEDBACK ------\n\n
                '''

            for i in range(len(prompts)):
                question_title = question_titles[i]
                prompt_output = prompt_outputs[i]
                shortform_feedback = shortform_feedbacks[i]
                longform_feedback = prompt_input_objects[i].get_longform_feedback(prompt_output)

                feedback += f'--- Q{i + 1}: {question_title} ---\n\n'
                feedback += f'Feedback {i + 1}: {shortform_feedback}\n\n'
                # if booleans_indicating_which_prompts_need_feedback[i] == True:
                feedback += f'Explanation {i + 1}: {longform_feedback}\n\n\n'

            feedback += f'--- Controlled risk multiplication is: {controlled_risk} ---\n\n'
            feedback += f'--- Uncontrolled risk multiplication is: {uncontrolled_risk} ---\n\n'

            for i in range(n_prompts):
                if booleans_indicating_which_prompts_need_feedback[i] == True:
                    st.write(prompt_outputs[i])
            
            if controlled_risk != 'correct':
                st.write(f'Controlled risk multiplication is: {controlled_risk}')

            if uncontrolled_risk != 'correct':
                st.write(f'Uncontrolled risk multiplication is: {uncontrolled_risk}')

            with st.expander('See Full Feedback'):
                st.write(feedback)
            
            if 'feedback' not in st.session_state:
                st.session_state.feedback = feedback

with st.form('feedback_from_user'):
    # slider_options = ['Strongly Disagree', 'Disagree', 'On the fence', 'Agree', 'Strongly Agree']
    options = ['Yes', 'No']
    st.write('Do you think the feedback received in this exercise is: ')
    is_feedback_correct = st.radio('i) correct?', options=options)
    why_not_correct = st.text_input('If you answered No, why is the feedback incorrect?')
    is_feedback_specific  = st.radio('ii) specific, i.e. tailored specifically to the answers you gave?', options=options)
    why_not_specific = st.text_input('If you answered No, why is the feedback not specific?')
    is_feedback_constructive = st.radio('iii) constructive, i.e. helped you understand why an answer was wrong and made helpful suggestions on how to improve your answer.', options=options)
    why_not_constructive = st.text_input('If you answered No, why is the feedback not constructive?')
    are_instructions_clear = st.radio('Do you think the instructions given for this exercise were clear?', options=options)
    why_not_clear = st.text_input('If you answered No, why are the instructions not clear?')
    learning_outcomes = st.radio('Do you feel you have obtained all the learning outcomes specified for this exercise?', options=options)
    why_not_learning_outcomes = st.text_input('If you answered No, why do you think you have not obtained all the learning outcomes?')
    better_prepared = st.radio('Do you feel better prepared for writing risk assessments in future after completing this exercise?', options=options)
    why_not_better_prepared = st.text_input('If you answered No, how do you think this exercise could be improved in general?')
    general_feedback = st.text_input('Any other general feedback?')
    submit_button = st.form_submit_button("Submit")
    # google_sheets_writer = GoogleSheetsWriter(spreadsheet_id=st.secrets["SPREADSHEET_ID"],
    #                                           secrets=st.secrets["gcp_service_account"])
    google_sheets_writer = GoogleSheetsWriter(sheet_name='Sheet1')

    if submit_button:
        if 'feedback' in st.session_state:
            google_sheets_writer.write_to_sheets(new_line_data=[
                is_feedback_correct,
                why_not_correct,
                is_feedback_specific,
                why_not_specific,
                is_feedback_constructive,
                why_not_constructive,
                are_instructions_clear,
                why_not_clear,
                learning_outcomes,
                why_not_learning_outcomes,
                better_prepared,
                why_not_better_prepared,
                general_feedback,
                st.session_state.feedback])
            st.write('Thank you for your feedback!')
        else:
            st.write('Please submit a risk assessment first')