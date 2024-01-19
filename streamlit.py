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