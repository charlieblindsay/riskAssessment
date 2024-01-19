import streamlit as st
from evaluation import evaluation_function
from GoogleSheetsWriter import GoogleSheetsWriter

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
            if 'feedback' not in st.session_state:
                st.session_state.feedback = result['feedback']
            # st.write(result)

with st.form('feedback_from_user'):
    name = st.text_input('Name', value='Charlie Lindsay')
    submit_button = st.form_submit_button("Submit")
    # google_sheets_writer = GoogleSheetsWriter(spreadsheet_id=st.secrets["SPREADSHEET_ID"],
    #                                           secrets=st.secrets["gcp_service_account"])
    google_sheets_writer = GoogleSheetsWriter(sheet_name='Sheet1')

    if submit_button:
        if 'feedback' in st.session_state:
            google_sheets_writer.write_to_sheets(new_line_data=[name, st.session_state.feedback])
            st.write('Thank you for your feedback!')
        else:
            st.write('Please submit a risk assessment first')