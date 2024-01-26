import streamlit as st
from evaluation import evaluation_function
import pandas as pd
from GoogleSheetsWriter import GoogleSheetsWriter

st.title('Risk Assessment Exercise')
st.subheader('Learning Objectives')
'Before completing this exercise, please read the following learning objectives:'
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
with st.expander('See Additional Instructions'):
    st.markdown('''
                1. Read through the helpful resources below.
                2. Using what you have learnt, fill out the Risk Assessment fields. NOTE: You should only include one example in each field, e.g. for the 'Prevention' field, only list one prevention.
                3. Your answers will then be processed (which takes around 10 seconds).
                4. Afterwards, feedback will be given. You should use the feedback to improve your answers.
                5. Once you have gotten everything correct, you can give feedback to me by filling out the feedback form at the bottom. Many Thanks, Charlie''')
    
st.title('Helpful Resources')

st.subheader('Input Field Definitions')
'Please read the definitions of the input fields below:'
with st.expander('Click to see Input Field Definitions'):
    definitions = {
        'Field': ['Activity', 'Hazard', 'How it harms', 'Who it harms', 'Prevention', 'Mitigation', 'Likelihood', 'Severity', 'Risk', 'Uncontrolled Likelihood', 'Uncontrolled Severity', 'Uncontrolled Risk', 'Controlled Likelihood', 'Controlled Severity', 'Controlled Risk'],
        'Definition': [
            'Activity involving the hazard',
            'Dangerous phenomenon, object, human activity or condition.',
            'Detailed explanation of how the hazard can cause harm.',
            'Individuals or groups at risk of harm from the hazard.',
            'Action which reduces the likelihood of the hazard causing harm.',
            'Action which reduces the harm caused by the hazard.',
            'The probability that the hazard causes harm. Ranges from 1-5.',
            'The degree of harm that the hazard can cause. Ranges from 1-5.',
            'Calculated using Risk = Likelihood x Severity.',
            'Likelihood before prevention measure applied.',
            'Severity before mitigation measure applied.',
            'Risk before prevention/mitigation applied.',
            'Likelihood after prevention measure applied.',
            'Severity after mitigation measure applied.',
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
        'Example': ['Fluids laboratory', 'Ink spillage', 'Serious eye damage', 'Students', 2, 
                    2, 4, 'Wear safety glasses', 'Wash eyes with water', 1, 1, 1]
    }

    df = pd.DataFrame(example_risk_assessment)
    st.table(df)


st.title('Risk Assessment Fields')
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

    # TODO: Add try except for when they are not connected to the internet.
    if submit_button:
        with st.spinner('Getting Risk Assessment Feedback...'):
            response = [activity, hazard, who_it_harms, how_it_harms, uncontrolled_likelihood, 
                        uncontrolled_severity, uncontrolled_risk, prevention, mitigation, 
                        controlled_likelihood, controlled_severity, controlled_risk]
            result = evaluation_function(response=response, answer='', params='')
            input_check_feedback_message = result['input_check_feedback_message']
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
            
            if input_check_feedback_message != '':
                st.write(input_check_feedback_message)

                full_feedback = f'''{input_check_feedback_message}'''

            if input_check_feedback_message == '' and (controlled_risk != 'correct' or uncontrolled_risk != 'correct'):
                st.write(f'Controlled risk multiplication is {controlled_risk}')
                st.write(f'Uncontrolled risk multiplication is {uncontrolled_risk}')

                full_feedback = f'''Controlled risk multiplication is {controlled_risk}')
                Uncontrolled risk multiplication is {uncontrolled_risk}'''

            if input_check_feedback_message == '' and controlled_risk == 'correct' and uncontrolled_risk == 'correct':
                feedback_table = {
                    'Input field': ['Activity', 'Hazard & How it harms', 'Who it harms', 'Prevention', 'Mitigation',
                                    'All Severity, Likelihood and Risk Inputs'],
                    'Feedback': []
                }

                for i in range(len(booleans_indicating_which_prompts_need_feedback)):
                    boolean = booleans_indicating_which_prompts_need_feedback[i]
                    if boolean == True:
                        feedback_table['Feedback'].append('Incorrect')
                    else:
                        feedback_table['Feedback'].append('Correct')
                
                # Since already checked that controlled_risk and uncontrolled_risk are correct, can just add correct to the table
                # Hence, 'All Severity, Likelihood and Risk Inputs' are correct
                feedback_table['Feedback'].append('Correct')
                
                most_pertinent_feedback_table = {'Input field': [], 'Feedback': []}
                for i in range(len(feedback_table['Input field'])):
                    if feedback_table['Feedback'][i] == 'Correct':
                        most_pertinent_feedback_table['Input field'].append(feedback_table['Input field'][i])
                        most_pertinent_feedback_table['Feedback'].append(feedback_table['Feedback'][i])
                    if feedback_table['Feedback'][i] == 'Incorrect':
                        most_pertinent_feedback_table['Input field'].append(feedback_table['Input field'][i])
                        most_pertinent_feedback_table['Feedback'].append(feedback_table['Feedback'][i])

                        break # To only show first field that is incorrect
                
                prompt_fields = ['Activity', 'Hazard & How it harms', 'Who it harms', 'Prevention', 'Mitigation']
                st.title('Feedback')
                st.table(pd.DataFrame(most_pertinent_feedback_table))

                n_prompts = len(prompts)

                for i in range(n_prompts):
                    if booleans_indicating_which_prompts_need_feedback[i] == True:
                        st.subheader(f'Feedback for Input(s): {prompt_fields[i]}')
                        shortform_feedback = shortform_feedbacks[i]
                        longform_feedback = prompt_input_objects[i].get_longform_feedback(prompt_outputs[i])

                        st.markdown(f'''
                                    - {shortform_feedback}
                                    - Explanation: {longform_feedback}'
                                    - Please look at the definition of the {prompt_fields[i]} input field(s) and the example risk assessment for assistance.
                                    ''')
                        
                        break # To only show feedback for first field that is incorrect
                
                feedback = f'''
                    ------ FEEDBACK ------\n\n
                    '''
                
                full_feedback = f'''
                    ------ FULL FEEDBACK ------\n\n
                    '''
                
                for i in range(len(prompts)):
                    question_title = question_titles[i]
                    prompt_output = prompt_outputs[i]
                    shortform_feedback = shortform_feedbacks[i]
                    longform_feedback = prompt_input_objects[i].get_longform_feedback(prompt_output)

                    feedback += f'--- Q{i + 1}: {question_title} ---\n\n'
                    feedback += f'Feedback {i + 1}: {shortform_feedback}\n\n'
                    feedback += f'Explanation {i + 1}: {longform_feedback}\n\n\n'

                    full_feedback += f'--- Q{i + 1}: {question_title} ---\n\n'
                    full_feedback += f'Feedback {i + 1}: {shortform_feedback}\n\n'
                    full_feedback += f'Explanation {i + 1}: {prompt_outputs[i]}\n\n\n'

                feedback += f'--- Controlled risk multiplication is: {controlled_risk} ---\n\n'
                feedback += f'--- Uncontrolled risk multiplication is: {uncontrolled_risk} ---\n\n'

                with st.expander('See Full Feedback'):
                    st.write(feedback)
                
            if 'feedback' not in st.session_state:
                st.session_state.feedback = [full_feedback]
                # st.session_state.feedback = full_feedback
            else:
                st.session_state.feedback.append(full_feedback)

st.title('Your feedback for me')
with st.expander('Please fill out this form so I can improve the Exercise!'):
    with st.form('feedback_from_user'):
        # slider_options = ['Strongly Disagree', 'Disagree', 'On the fence', 'Agree', 'Strongly Agree']
        options = ['Yes', 'No']
        name = st.text_input('First Name')
        st.write('Do you think the feedback received in this exercise is: ')
        is_feedback_correct = st.radio('i) correct?', options=options)
        why_not_correct = st.text_input('If you answered No, why is the feedback incorrect?')
        is_feedback_specific  = st.radio('ii) specific, i.e. tailored specifically to the answers you gave?', options=options)
        why_not_specific = st.text_input('If you answered No, why is the feedback not specific?')
        is_feedback_constructive = st.radio('iii) constructive, i.e. helped you understand why an answer was wrong and made helpful suggestions on how to improve your answer.', options=options)
        why_not_constructive = st.text_input('If you answered No, why is the feedback not constructive?')
        is_feedback_concise = st.radio('iv) concise?', options=options)
        why_not_concise = st.text_input('If you answered No, why is the feedback not concise?')
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
                    '\n\n\n'.join(st.session_state.feedback),
                    is_feedback_correct,
                    why_not_correct,
                    is_feedback_concise,
                    why_not_concise,
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
                    general_feedback
                    ])
                st.write('Thank you very much for your feedback! It will be taken on board!')
            else:
                st.write('Please submit a risk assessment first')