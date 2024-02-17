import re

class RegexPatternMatcher:
    def __init__(self):
        pass
    
    def check_string_for_true_or_false(self, string):
        pattern = re.compile(r"Overall Answer: (true|false)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower() == "true"
        else:
            raise Exception("Pattern not found in output prompt")
    
    def check_string_for_true_or_false_with_no_overall_answer(self, string):
        pattern = re.compile(r"Answer: (true|false)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower() == "true"
        else:
            raise Exception("Pattern not found in output prompt")
        
    def check_string_for_prevention_mitigation_or_neither(self, string):
        pattern = re.compile(r"Answer: (prevention|mitigation|neither|both)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower()
        else:
            raise Exception("Pattern not found in output prompt")
    
    def check_string_for_type_of_input_field(self, string):
        pattern = re.compile(r"Answer: (activity|hazard|event that leads to harm|harm caused|who|control measure)", re.IGNORECASE)
        match = re.search(pattern, string)
        if match:
            return match.group(1).lower()
        else:
            raise Exception("Pattern not found in output prompt")
        
    def extract_section_of_prompt_until_new_line_or_end_of_string(self, prompt_output, prompt_section_title):
        pattern = rf"{prompt_section_title}:\s*(.*?)(?=\n|$)"
        match = re.search(pattern, prompt_output, re.DOTALL)
        if match:
            explanation = match.group(1).strip()
            return explanation
        else:
            raise Exception("Pattern not found in output prompt")
        
    def always_return_false(self, prompt_output):

        return False
    
    def always_return_true(self, prompt_output):
        
        return True
    
    def extract_illness(self, prompt_output):
        if self.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, "Illness") == "False":
            return False
        else:
            illness = self.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, "Illness")
            return illness
    
    def extract_injury(self, prompt_output):
        if self.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, "Injury") == "False":
            return False
        else:
            injury = self.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, "Injury")
            return injury
        
    def extract_hazard_event(self, prompt_output):
    
        hazard_event = self.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, "Event that leads to harm")
        
        return hazard_event