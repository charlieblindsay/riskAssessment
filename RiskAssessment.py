from typing import Type
import csv


try:
    from .PromptInputs import *
    from .LLMCaller import *
    from .RegexPatternMatcher import RegexPatternMatcher
except ImportError:
    from PromptInputs import *
    from LLMCaller import *
    from RegexPatternMatcher import RegexPatternMatcher

class RiskAssessment:
    def __init__(self, activity, hazard, who_it_harms, how_it_harms,
                  uncontrolled_likelihood, uncontrolled_severity, uncontrolled_risk,
                 prevention, mitigation, controlled_likelihood, controlled_severity, controlled_risk,
                 prevention_prompt_expected_output, mitigation_prompt_expected_output):
        self.activity = activity
        self.hazard = hazard
        self.who_it_harms = who_it_harms
        self.how_it_harms = how_it_harms
        self.uncontrolled_likelihood = uncontrolled_likelihood
        self.uncontrolled_severity = uncontrolled_severity
        self.uncontrolled_risk = uncontrolled_risk
        self.prevention = prevention
        self.mitigation = mitigation
        self.controlled_likelihood = controlled_likelihood
        self.controlled_severity = controlled_severity
        self.controlled_risk = controlled_risk

        self.prevention_prompt_expected_output = prevention_prompt_expected_output
        self.mitigation_prompt_expected_output = mitigation_prompt_expected_output

    def get_word_fields(self):
        return ['activity',
                'hazard',
                'who_it_harms',
                'how_it_harms',
                'prevention',
                'mitigation']
    
    def get_integer_fields(self):
        return ['uncontrolled_likelihood',
                'uncontrolled_severity',
                'uncontrolled_risk',
                'controlled_likelihood',
                'controlled_severity',
                'controlled_risk']
    
    def get_empty_fields(self):
        empty_fields = []

        for field_name in self.get_word_fields() + self.get_integer_fields():
            if getattr(self, field_name) == '':
                formatted_field_name = field_name.replace('_', ' ').title()
                empty_fields.append(formatted_field_name)
        
        return empty_fields
        
    def does_string_represent_an_integer(self, string:str):
        try:
            int(string)
            return True
        except ValueError:
            return False
        
    def does_string_represent_words(self, string:str):
        if len(string.split(' ')) > 1: # If there are multiple words, unlikely to represent a number
            return True
        else:
            return string.isalpha() # If there is only one word, check if it is a word and not a number
        
    def get_word_fields_incorrect(self):
        word_fields_incorrect = []

        for word_field_name in self.get_word_fields():
            if not self.does_string_represent_words(getattr(self, word_field_name)):
                formatted_word_field_name = word_field_name.split('_', ' ').title()
                word_fields_incorrect.append(formatted_word_field_name)
        
        return word_fields_incorrect
    
    def get_integer_fields_incorrect(self):
        integer_fields_incorrect = []

        for integer_field_name in self.get_integer_fields():
            if not self.does_string_represent_an_integer(getattr(self, integer_field_name)):
                formatted_integer_field_name = integer_field_name.replace('_', ' ').title()
                integer_fields_incorrect.append(formatted_integer_field_name)
        
        return integer_fields_incorrect
    
    def get_input_check_feedback_message(self):
        empty_fields = self.get_empty_fields()
        word_fields_incorrect = self.get_word_fields_incorrect()
        integer_fields_incorrect = self.get_integer_fields_incorrect()
        
        feedback_message = ''

        if len(empty_fields) > 0:
            feedback_message += f'Please fill in the following fields: {str(empty_fields)[1:-1]}.\n\n'
        
        if len(empty_fields) == 0 and len(word_fields_incorrect) > 0:
            feedback_message += f'Please make sure that the following fields only contain words: {str(word_fields_incorrect)[1:-1]}.\n\n'
        
        if len(empty_fields) == 0 and len(integer_fields_incorrect) > 0:
            feedback_message += f'Please make sure that the following fields are a single integer: {str(integer_fields_incorrect)[1:-1]}.\n\n'
        
        return feedback_message

    def convert_RiskAssessment_object_into_lambda_response_list(self):
        return list(vars(self).values())
    
    def get_activity_input(self):
        return Activity(activity=self.activity)
    
    def get_how_it_harms_in_context_input(self):
        return HowItHarmsInContext(how_it_harms=self.how_it_harms,
                          activity=self.activity,
                          hazard = self.hazard)
    
    def get_who_it_harms_in_context_input(self):
        return WhoItHarmsInContext(who_it_harms=self.who_it_harms,
                            activity=self.activity)
    
    def get_prevention_input(self):
        return Prevention(prevention=self.prevention,
                          activity=self.activity,
                          hazard=self.hazard,
                          how_it_harms=self.how_it_harms,
                          who_it_harms=self.who_it_harms)
    
    def get_mitigation_input(self):
        return Mitigation(mitigation=self.mitigation,
                          activity=self.activity,
                          hazard=self.hazard,
                          how_it_harms=self.how_it_harms,
                          who_it_harms=self.who_it_harms)
    
    def check_that_risk_equals_likelihood_times_severity(self, likelihood, severity, risk):
        try:
            likelihood = int(likelihood)
            severity = int(severity)
            risk = int(risk)

            if likelihood * severity == risk:
                return 'correct'
            else:
                return 'incorrect. Please check your multiplication.'
        
        except ValueError:
            return 'Please make sure that the likelihood, severity, and risk are all integers.'

    def check_uncontrolled_risk(self):
        return self.check_that_risk_equals_likelihood_times_severity(self.uncontrolled_likelihood,
                                                                self.uncontrolled_severity,
                                                                self.uncontrolled_risk)
    
    def check_controlled_risk(self):
        return self.check_that_risk_equals_likelihood_times_severity(self.controlled_likelihood,
                                                                self.controlled_severity,
                                                                self.controlled_risk)
    
    # TODO: Add ability to see prompt output percentages - might be possible for LLMs other than GPT-3
    
    def get_list_of_prompt_input_objects(self):
        return [self.get_activity_input(),
                self.get_how_it_harms_in_context_input(),
                self.get_who_it_harms_in_context_input(),
                self.get_prevention_input(),
                self.get_mitigation_input()
                ]
    
    def get_list_of_question_titles(self):
        question_titles = []
        
        for prompt_input_object in self.get_list_of_prompt_input_objects():
            question_titles.append(prompt_input_object.get_question_title())

        return question_titles
    
    def get_list_of_prompts(self):
        prompts = []

        for prompt_input_object in self.get_list_of_prompt_input_objects():
            prompts.append(prompt_input_object.generate_prompt())

        return prompts
    
    def get_list_of_prompt_outputs(self, LLM_caller: Type[LLMCaller]):
        prompt_outputs = []

        for prompt_input_object in self.get_list_of_prompt_input_objects():
            prompt_outputs.append(LLM_caller.get_model_output(prompt_input_object))
        
        return prompt_outputs
    
    def get_list_of_regex_matches(self, prompt_outputs):
        regex_pattern_matcher = RegexPatternMatcher()

        regex_matches = []

        prompt_inputs = self.get_list_of_prompt_input_objects()

        for i in range(len(prompt_inputs)):
            pattern_matching_method = getattr(regex_pattern_matcher, prompt_inputs[i].pattern_matching_method)

            regex_match = pattern_matching_method(prompt_outputs[i])
            regex_matches.append(regex_match)
        
        return regex_matches
    
    def get_list_of_shortform_feedback_objects(self):
        shortform_feedback_objects = []

        for prompt_input_object in self.get_list_of_prompt_input_objects():
            shortform_feedback_objects.append(prompt_input_object.get_shortform_feedback())

        return shortform_feedback_objects

    def get_list_of_shortform_feedback_from_regex_matches(self, regex_matches):
        list_of_shortform_feedback = []

        prompt_inputs = self.get_list_of_prompt_input_objects()

        shortform_feedback_objects = self.get_list_of_shortform_feedback_objects()

        for i in range(len(regex_matches)):
            if regex_matches[i] in prompt_inputs[i].correct_matched_patterns:
                list_of_shortform_feedback.append(shortform_feedback_objects[i].positive_feedback)
            else:
                list_of_shortform_feedback.append(shortform_feedback_objects[i].negative_feedback)
            
        return list_of_shortform_feedback
    
    def get_booleans_indicating_which_prompts_need_feedback(self, regex_matches):
        booleans_indicating_which_prompts_need_feedback = []

        prompt_inputs = self.get_list_of_prompt_input_objects()

        for i in range(len(regex_matches)):
            if regex_matches[i] in prompt_inputs[i].correct_matched_patterns:
                booleans_indicating_which_prompts_need_feedback.append(False)
            else:
                booleans_indicating_which_prompts_need_feedback.append(True)
            
        return booleans_indicating_which_prompts_need_feedback
    
    def are_all_multiplications_correct(self)->bool:
        return self.check_uncontrolled_risk() == 'correct' and self.check_controlled_risk() == 'correct'
    
    def are_all_prompt_outputs_correct(self, prompt_outputs) -> bool:
        regex_matches = self.get_list_of_regex_matches(prompt_outputs)
        prompt_inputs = self.get_list_of_prompt_input_objects()

        for i in range(len(regex_matches)):
            if regex_matches[i] != prompt_inputs[i].correct_matched_patterns:
                return False
        
        return True