from typing import Type

try:
    from .PromptInputs import *
    from .LLMCaller import *
    from .RegexPatternMatcher import RegexPatternMatcher
except ImportError:
    from PromptInputs import *
    from LLMCaller import *
    from RegexPatternMatcher import RegexPatternMatcher

class RiskAssessment:
    def __init__(self, activity, hazard, how_it_harms, who_it_harms,
                  uncontrolled_likelihood, uncontrolled_severity, uncontrolled_risk,
                 prevention, mitigation, controlled_likelihood, controlled_severity, controlled_risk,
                 risk_domain, prevention_prompt_expected_output, mitigation_prompt_expected_output):
        self.activity = activity
        self.hazard = hazard
        self.how_it_harms = how_it_harms
        self.who_it_harms = who_it_harms
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
        self.risk_domain = risk_domain

        # TODO: Remove these parameters
        self.always_true = True
        self.always_false = False

    def to_string(self):
        class_name = self.__class__.__name__
        if hasattr(self, '__dict__'):
            attributes = ', '.join([f"{key}={value}" for key, value in self.__dict__.items()])
            return f"{class_name}({attributes})"
        else:
            return f"{class_name}()"

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
    
    # TODO: It should be OK if either prevention or mitigation field is empty, but not both.
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
                formatted_word_field_name = word_field_name.replace('_', ' ').title()
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

        # TODO: Should make it non-compulsory to enter anything for the mitigation field
        if len(empty_fields) > 0:
            feedback_message += f'Please fill in the following fields: {str(empty_fields)[1:-1]}.\n\n'
        
        if len(empty_fields) == 0 and len(word_fields_incorrect) > 0:
            feedback_message += f'Please make sure that the following fields only contain words: {str(word_fields_incorrect)[1:-1]}.\n\n'
        
        if len(empty_fields) == 0 and len(integer_fields_incorrect) > 0:
            feedback_message += f'Please make sure that the following fields are a single integer: {str(integer_fields_incorrect)[1:-1]}.\n\n'
        
        return feedback_message

    def convert_RiskAssessment_object_into_lambda_response_list(self):
        return list(vars(self).values())
    
    def get_no_information_provided_for_prevention_input(self):
        return NoInformationProvided(input=self.prevention)
    
    def get_no_information_provided_for_mitigation_input(self):
        return NoInformationProvided(input=self.mitigation)
    
    def get_activity_input(self):
        return Activity(activity=self.activity)

    def get_how_it_harms_in_context_input(self):
        return HowItHarmsInContext(how_it_harms=self.how_it_harms,
                          activity=self.activity,
                          hazard = self.hazard)
    
    def get_who_it_harms_in_context_input(self):
        return WhoItHarmsInContext(who_it_harms=self.who_it_harms,
                            activity=self.activity,
                            hazard=self.hazard,
                            how_it_harms=self.how_it_harms)
    
    def get_harm_caused_and_hazard_event_input(self):
        return HarmCausedAndHazardEvent(
                            activity=self.activity,
                            hazard=self.hazard,
                          how_it_harms=self.how_it_harms,
                            who_it_harms=self.who_it_harms)
    
    def get_control_measure_prompt_with_prevention_input(self):
        return PreventionPrompt(
            activity=self.activity,
            who_it_harms=self.who_it_harms,
            how_it_harms=self.how_it_harms,
            hazard=self.hazard,
            control_measure=self.prevention)
    
    def get_control_measure_prompt_with_mitigation_input(self):
        return MitigationPrompt(
            activity=self.activity,
            who_it_harms=self.who_it_harms,
            how_it_harms=self.how_it_harms,
            hazard=self.hazard,
            control_measure=self.mitigation)
    
    def is_future_harm_reduced_prompt_input_with_prevention(self):
        return IsFutureHarmReduced(
            activity=self.activity,
            control_measure=self.prevention,
            who_it_harms = self.who_it_harms
        )
    
    def is_future_harm_reduced_prompt_input_with_mitigation(self):
        return IsFutureHarmReduced(
            activity=self.activity,
            control_measure=self.mitigation,
            who_it_harms = self.who_it_harms
        )
    
    def get_risk_domain_classification_input(self):
        return RiskDomainClassification(hazard=self.hazard,
                                        how_it_harms=self.how_it_harms,
                                        who_it_harms=self.who_it_harms)
    
    def check_that_likelihood_and_severity_values_are_between_1_and_4(self, likelihood, severity):
        try:
            likelihood = int(likelihood)
            severity = int(severity)

            if likelihood > 4 or likelihood < 1 or severity > 4 or severity < 1:
                return 'Incorrect. Please make sure that, as per the Risk Assessment convention shown in Question 1, all of the likelihood and severity values are between 1 and 4.'
            else:
                return 'correct'
            
        except ValueError:
            return 'Please make sure that the likelihood and severity are both integers.'
        
    def check_that_uncontrolled_likelihood_and_severity_values_are_between_1_and_4(self):
        return self.check_that_likelihood_and_severity_values_are_between_1_and_4(self.uncontrolled_likelihood, self.uncontrolled_severity)
    
    def check_that_controlled_likelihood_and_severity_values_are_between_1_and_4(self):
        return self.check_that_likelihood_and_severity_values_are_between_1_and_4(self.controlled_likelihood, self.controlled_severity)

    def check_that_risk_equals_likelihood_times_severity(self, likelihood, severity, risk):
        try:
            likelihood = int(likelihood)
            severity = int(severity)
            risk = int(risk)

            if likelihood * severity == risk:
                return 'correct'
            else:
                return 'incorrect. Please check that your risk values are equal to the likelihood multiplied by the severity.'
            
        except ValueError:
            return 'Please make sure that the likelihood, severity, and risk are all integers.'

    def check_uncontrolled_risk_multiplication(self):
        return self.check_that_risk_equals_likelihood_times_severity(self.uncontrolled_likelihood,
                                                                self.uncontrolled_severity,
                                                                self.uncontrolled_risk)
    
    def check_controlled_risk_multiplication(self):
        return self.check_that_risk_equals_likelihood_times_severity(self.controlled_likelihood,
                                                                self.controlled_severity,
                                                                self.controlled_risk)
    
    def check_controlled_values_are_less_than_or_equal_to_uncontrolled_values(self, controlled_value, uncontrolled_value, value_name):
        try:
            controlled_value = int(controlled_value)
            uncontrolled_value = int(uncontrolled_value)

            if controlled_value <= uncontrolled_value:
                return 'correct'
            else:
                return f'Your {value_name} values are incorrect. Please make sure that the controlled {value_name} is less than or equal to the uncontrolled {value_name}.'
            
        except ValueError:
            return 'Please make sure that the controlled and uncontrolled values are both integers.'
    
    def compare_controlled_and_uncontrolled_likelihood(self):
        return self.check_controlled_values_are_less_than_or_equal_to_uncontrolled_values(controlled_value=self.controlled_likelihood,
                                                                                           uncontrolled_value=self.uncontrolled_likelihood,
                                                                                           value_name='likelihood')

    def compare_controlled_and_uncontrolled_severity(self):
        return self.check_controlled_values_are_less_than_or_equal_to_uncontrolled_values(controlled_value=self.controlled_severity,
                                                                                           uncontrolled_value=self.uncontrolled_severity,
                                                                                           value_name='severity')
    
    # TODO: Add ability to see prompt output percentages - might be possible for LLMs other than GPT-3

    def get_list_of_input_field_classification_prompt_input_objects(self):
        return [self.activity_field_classification_input(),
                self.hazard_field_classification_input(),
                self.how_it_harms_field_classification_input(),
                self.who_it_harms_field_classification_input(),
                self.get_prevention_field_classification_input(),
                self.get_mitigation_field_classification_input()]

    def get_prompt_output_and_pattern_matched(self, prompt_input_object: Type[PromptInput], LLM_caller: Type[LLMCaller], **kwargs):
        regex_pattern_matcher = RegexPatternMatcher()
        
        prompt_output = LLM_caller.get_model_output(prompt_input_object.generate_prompt(**kwargs))
        print(prompt_output)
        
        pattern_matching_method = getattr(regex_pattern_matcher, prompt_input_object.pattern_matching_method)
        
        pattern_matched = pattern_matching_method(prompt_output)

        return prompt_output, pattern_matched
    
    def get_shortform_feedback_from_regex_match(self, prompt_input_object: Type[PromptInput], pattern_matched):
        
        if pattern_matched in prompt_input_object.labels_indicating_correct_input:
            return prompt_input_object.get_shortform_feedback(feedback_type='positive')
        else:
            return prompt_input_object.get_shortform_feedback(feedback_type='negative')
    
    def are_all_multiplications_correct(self)->bool:
        return self.check_uncontrolled_risk() == 'correct' and self.check_controlled_risk() == 'correct'
    
    def are_all_prompt_outputs_correct(self, prompt_outputs) -> bool:
        regex_matches = self.get_list_of_regex_matches(prompt_outputs)
        prompt_inputs = self.get_list_of_prompt_input_objects()

        for i in range(len(regex_matches)):
            if regex_matches[i] != prompt_inputs[i].labels_indicating_correct_input:
                return False
        
        return True

class RiskAssessmentWithoutNumberInputs(RiskAssessment):
    def __init__(self, activity, hazard, how_it_harms, who_it_harms,
                 prevention, mitigation, 
                 risk_domain,
                 prevention_prompt_expected_output, 
                 mitigation_prompt_expected_output):
        super().__init__(activity=activity,
                         hazard=hazard,
                         how_it_harms=how_it_harms,
                         who_it_harms=who_it_harms,
                         uncontrolled_likelihood='1',
                         uncontrolled_severity='1',
                         uncontrolled_risk='1',
                         prevention=prevention,
                         mitigation=mitigation,
                         controlled_likelihood='1',
                         controlled_severity='1',
                         controlled_risk='1',
                         risk_domain=risk_domain,
                         prevention_prompt_expected_output=prevention_prompt_expected_output,
                         mitigation_prompt_expected_output=mitigation_prompt_expected_output)