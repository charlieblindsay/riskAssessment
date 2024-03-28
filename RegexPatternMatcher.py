import re

class HarmCausedAndHazardEventAndHazardEvent:
    def __init__(self, harm_caused, hazard_event):
        self.harm_caused = harm_caused
        self.hazard_event = hazard_event

class RegexPatternMatcher:
    def __init__(self):
        pass
    
    def check_string_for_true_or_false(self, string):
        pattern = re.compile(r"Overall Answer: (true|false)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower() == "true"
        else:
            print(string)
            return 'No pattern found'
    
    def check_string_for_true_or_false_with_no_overall_answer(self, string):
        pattern = re.compile(r"Answer: (true|false)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower() == "true"
        else:
            return 'No pattern found'
        
    def check_string_for_no_information_provided(self, string):
        pattern = re.compile(r"Answer: (control measure|no information provided)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower()
        else:
            return 'No pattern found'
        
    def check_string_for_classification(self, string):
        pattern = re.compile(r"Classification: (physical risk to individuals|environmental risk)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower()
        else:
            return 'No pattern found'
        
    def check_string_for_prevention_mitigation_or_neither(self, string):
        pattern = re.compile(r"Answer: (prevention|mitigation|neither|both)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower()
        else:
            return 'No pattern found'
    
    def check_string_for_type_of_input_field(self, string):
        pattern = re.compile(r"Answer: (activity|hazard|event that leads to harm|harm caused|who|control measure)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower()
        else:
            return 'No pattern found'
        
    def extract_section_of_prompt_until_new_line_or_end_of_string(self, prompt_output, prompt_section_title):
        pattern = rf"{prompt_section_title}:\s*(.*?)(?=\n|$)"
        match = re.search(pattern, prompt_output, re.DOTALL)
        if match:
            explanation = match.group(1).strip()
            return explanation
        else:
            return 'No pattern found'
        
    def extract_harm_caused_and_hazard_event(self, prompt_output):

        harm_caused = self.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, "Harm caused")
        hazard_event = self.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, "Event that leads to harm")
        
        return HarmCausedAndHazardEventAndHazardEvent(harm_caused=harm_caused, hazard_event=hazard_event)
    
    def extract_overall_answer_until_end_of_line(self, prompt_output):

        return self.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, "Overall Answer")
    
    def always_return_false(self, prompt_output):

        return False
    
    def always_return_true(self, prompt_output):
        
        return True