# TODO: Decide whether to remove the get_question method from the PromptInput class.
from RegexPatternMatcher import RegexPatternMatcher

class ShortformFeedback:
    def __init__(self, positive_feedback, negative_feedback):
        self.positive_feedback = positive_feedback
        self.negative_feedback = negative_feedback

class PromptInput:
    def __init__(self):
        self.activity_definition = """an action or process that involves
        physical or mental effort."""

        self.hazard_definition = """a dangerous phenomenon, substance, human activity or condition. 
        It may cause loss of life, injury or other health impacts, property damage, loss of livelihoods 
        and services, social and economic disruption, or environmental damage."""

        self.how_it_harms_entry_definition = """
        the potential negative consequences of a hazard. It can outline the specific impacts on
        human health, property, environment, economics, social structures, livelihoods, essential 
        services, and the risk of loss of life. It must be specific, clear and precise."""

        self.who_it_harms_entry_definition = """
        specific individuals, groups, environmental components or infrastructure
        likely to be negatively affected by identified risks, 
        excluding abstract concepts, generic terms or vague terms."""

        self.prevention_definition = f'an action which directly reduces the probability that the hazard occurs.'

        # TODO: I changed the definition of mitigation. See if this has an effect.
        self.mitigation_definition = f'''an action which directly reduces the harm caused by a hazard occurring
        or reduces the harm caused by the hazard after it has occurred.''' 

        self.pattern_matching_method = 'check_string_for_true_or_false'
        self.correct_matched_pattern = True

    def get_question(self):
        pass

    def get_question_title(self):
        pass

    def generate_prompt(self):
        pass

    def get_shortform_feedback(self):
        pass

    # Using regular expressions, extracts the relevant information from the prompt output.
    def get_longform_feedback(self, prompt_output):
        pass

    def to_string(self):
        class_name = self.__class__.__name__
        if hasattr(self, '__dict__'):
            attributes = ', '.join([f"{key}={value}" for key, value in self.__dict__.items()])
            return f"{class_name}({attributes})"
        else:
            return f"{class_name}()"

class Activity(PromptInput):
    def __init__(self, activity: str):
        super().__init__()
        self.activity = activity

    def get_question_title(self):
        return 'Activity'

    def get_question(self):
        return f'''Is the 'activity': '{self.activity}' correct?'''

    def generate_prompt(self):
        return f'''
        An 'activity' is defined as """{self.activity_definition}""".

        Follow these instructions:
        1. In one sentence, provide a description of "{self.activity}". 
        2. In one sentence, compare this description with the provided definition of an activity. 
        3. If "{self.activity}" is an activity, answer True, else answer False. 
        
        Use the following output format:
        Description: <your description>
        Comparison: <your comparison>
        Answer: <your answer>'''
    
    def get_shortform_feedback(self):
        return ShortformFeedback(positive_feedback=f"Correct! '{self.activity}' is an activity.",
                                 negative_feedback=f"Incorrect. '{self.activity}' is not an activity.")
    
    def get_longform_feedback(self, prompt_output):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Comparison', 'Answer')
    
class HowItHarmsInContext(PromptInput):
    def __init__(self, how_it_harms, activity, hazard):
        super().__init__()
        self.how_it_harms = how_it_harms
        self.activity = activity
        self.hazard = hazard

    def get_question_title(self):
        return 'How It Harms'

    def get_question(self):
        return f'''Is 'how it harms': '{self.how_it_harms}' a way that the 'hazard': '{self.hazard}' 
        during the 'activity': '{self.activity}' causes harm?'''
    
    # TODO: Update this prompt in the main repo
    # TODO: Scope for adding a 'Add more detail' output.
    
    def generate_prompt(self):
        example_of_correct_how_it_harms = '''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Wet hands' during the
        activity: 'Fluids laboratory' given how the hazard harms: 
        'Electrocution'.
        2. In one sentence, explain whether or not 'Electrocution' 
        is a way that the hazard: 'Wet hands' causes harm.
        3. If 'Electrocution' is a way that the hazard: 
        'Wet hands' causes harm, answer True, else answer False.

        Output:
        1. Description: It is argued that wet hands during a fluids laboratory can cause harm through electrocution.
        2. Explanation: As water is a conductor of electricity, touching electronics with wet hands can cause electrocution as
        the water provides a path for electrical current to flow through the body.
        3. Answer: True
        '''

        example_of_incorrect_how_it_harms = '''
        Follow these instructions:
        1. In one sentence, describe the hazard: "Ink spillage" during the
        activity: "Fluids laboratory".
        2. In one sentence, explain whether or not "Radiation exposure" is a way that this hazard causes harm.
        3. If the hazard causes harm, answer True, else, answer False.
        
        Output:
        1. Description: It is argued that an ink spillage during a fluids laboratory can cause radiation exposure.
        2. Explanation: Radiation exposure is not a way that ink spillage during the fluids laboratory causes harm, 
        as the hazard primarily involves physical contamination rather than radiation.
        3. Answer: False.
        '''
        return f'''
        {example_of_correct_how_it_harms}

        {example_of_incorrect_how_it_harms}

        Follow these instructions:
        1. In one sentence, describe the hazard: "{self.hazard}" during the 
        activity: "{self.activity}".
        2. In one sentence, explain whether or not "{self.how_it_harms}" is a way that this hazard causes harm. 
        3. If the hazard causes harm, answer True, else, answer False.

        Use the following output format:
        Description: <your description>
        Explanation: <your Explanation>
        Answer: <your answer>'''
    
    def get_shortform_feedback(self):
        return ShortformFeedback(positive_feedback=f"Correct! '{self.how_it_harms}' is a way that the hazard: '{self.hazard}' causes harm.",
        negative_feedback=f"Incorrect. '{self.how_it_harms}' is not a way that the hazard: '{self.hazard}' causes harm.")
    
    def get_longform_feedback(self, prompt_output):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Explanation', 'Answer')
    
class WhoItHarmsInContext(PromptInput):
    def __init__(self, who_it_harms, how_it_harms, activity, hazard):
        super().__init__()
        self.who_it_harms = who_it_harms
        self.how_it_harms = how_it_harms
        self.activity = activity
        self.hazard = hazard

    def get_question_title(self):
        return 'Who It Harms'

    def get_question(self):
        return f'''Could 'who it harms': '{self.who_it_harms}' 
        be harmed by the 'hazard': '{self.hazard}' during 'activity': '{self.activity}'
        given how the hazard harms: '{self.how_it_harms}'?'''

    def generate_prompt(self):
        return f'''Follow these instructions:
        1. In one sentence, describe the hazard: '{self.hazard}' during the 
        activity: '{self.activity}' and how it harms: '{self.how_it_harms}'.
        2. In one sentence, explain whether or not 'who it harms': '{self.who_it_harms}' is harmed by this hazard. 
        3. If 'who it harms' is harmed by this hazard, answer True, else answer False.

        Your answer should be in the format:
        Description: <your description>
        Explanation: your_explanation
        Answer: <your answer>'''
    
    def get_shortform_feedback(self):
        return ShortformFeedback(positive_feedback=f"Correct! '{self.who_it_harms}' could be harmed by the hazard: '{self.hazard}'.",
        negative_feedback=f"Incorrect. '{self.who_it_harms}' could not be harmed by the hazard: '{self.hazard}'.")

    def get_longform_feedback(self, prompt_output):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Explanation', 'Answer')
    
class Prevention(PromptInput):
    def __init__(self, prevention, activity, hazard, how_it_harms, who_it_harms):
        super().__init__()
        self.prevention = prevention
        self.activity = activity
        self.hazard = hazard
        self.how_it_harms = how_it_harms
        self.who_it_harms = who_it_harms

        self.pattern_matching_method = 'check_string_for_prevention_mitigation_or_neither'
        self.correct_matched_pattern = 'prevention'

    def get_question_title(self):
        return 'Prevention'
    
    def get_question(self):
        return f'''Will the prevention measure: '{self.prevention}' reduce the likelihood of the
        'hazard': '{self.hazard}' occurring during the 'activity': {self.activity}, given
        given how the hazard harms: '{self.how_it_harms}' and who/what the hazard harms: '{self.who_it_harms}?'''

    def generate_prompt(self):
        example_of_correct_prevention = '''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Wet hands' during the
        activity: 'Fluids laboratory' given how the hazard harms: 'Electric shock of students when touching electronics (pump power supply) with wet hands'
        and who/what the hazard harms: 'Students'.
        2. Explain whether or not 'Students should make sure they touch electronics only with dry hands' reduces the likelihood of the hazard described in step 1.
        If so, it is a prevention measure.
        3. Assuming the hazard described in step 1 has led to harm, explain whether or not 'Students should make sure they touch electronics only with dry hands'
        would reduce or remove the harm caused by the hazard described in step 1.
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        prevention measure and a mitigation measure, answer 'both'.
        
        Output:
        Description: The hazard of 'Wet hands' during the activity 'Fluids laboratory' can lead to electric shock of students when 
        touching electronics (pump power supply) with wet hands.
        Prevention Explanation: 'Students should make sure they touch electronics only with dry hands' is a prevention measure reduces the probability 
        they will have wet hands and that an electric shock due to wet hands occurs. It is therefore a prevention measure.
        Mitigation Explanation: Assuming the hazard of 'Wet hands' has led to the harm of an electric shock, 
        'Students should make sure they touch electronics only with dry hands' does not directly reduce the severity of this harm
        as the electric shock has already occurred and touching the electronics with dry hands will not reverse this harm.
        It is therefore not a mitigation measure.
        Answer: Prevention'''

        example_of_mitigation = '''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Ink spillage' during the
        activity: 'Fluids laboratory' given how the hazard harms: 'Serious eye damage'
        and who/what the hazard harms: 'Students'.
        2. Explain whether or not 'Wash your eyes with clean water' reduces the likelihood of the hazard described in step 1.
        If so, it is a prevention measure.
        3. Assuming the hazard described in step 1 has led to harm, explain whether or not 'Wash your eyes with clean water'
        would reduce or remove the harm caused by the hazard described in step 1.
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        prevention measure and a mitigation measure, answer 'both'.

        Output: 
        Description: The hazard of 'Ink spillage' during the activity 'Fluids laboratory' can lead to serious eye damage to students.
        Prevention Explanation: 'Wash your eyes with clean water' does not reduce the likelihood that
        an ink spillage will lead to eye damage; it is a response or first aid action. Therefore, it is not a prevention.
        Mitigation Explanation: Assuming the hazard of ink spillage has led to the harm of serious eye damage, 
        'Wash your eyes with clean water' will reduce provide initial care and reduce the potential severity of this eye damage.
        It is therefore a mitigation. 
        Answer: Mitigation.'''

        example_of_incorrect_prevention = '''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Exposure to toxic welding fumes' during the
        activity: 'Welding metal structures' given how the hazard harms: 'Inhaling welding fumes can lead to respiratory problems, lung damage, and long-term health issues.'
        and who/what the hazard harms: 'Welders and individuals in the vicinity of the welding area.'.    
        2. Explain whether or not 'Using the welding equipment in an enclosed space without proper ventilation.' reduces the likelihood of the hazard described in step 1.
        If so, it is a prevention measure.
        3. Assuming the hazard described in step 1 has led to harm, explain whether or not 'Using the welding equipment in an enclosed space without proper ventilation.'
        would reduce or remove the harm caused by the hazard described in step 1.
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        prevention measure and a mitigation measure, answer 'both'.

        Output:
        Description: The hazard of 'Exposure to toxic welding fumes' during the activity 'Welding metal structures' can lead to 
        inhaling welding fumes, resulting in respiratory problems, lung damage, and long-term health issues for welders and individuals
        in the vicinity of the welding area.
        Prevention Explanation: 'Using the welding equipment in an enclosed space without proper ventilation' 
        will not reduce the likelihood of welders being exposed to toxic welding fumes and actually increases the likelihood of exposure.
        It is therefore not a prevention. 
        Mitigation Explanation: Assuming the hazard of toxic welding fumes exposure has led to respiratory problems, 
        'Using the welding equipment in an enclosed space without proper ventilation' will not serve to releive these respiratory problems
        and hence the severity of the harm is not reduced. It is therefore not a mitigation measure. 
        Answer: neither'''

        return f'''
        {example_of_correct_prevention}

        {example_of_mitigation}

        {example_of_incorrect_prevention}
        
        Follow these instructions:
        1. In one sentence, describe the hazard: '{self.hazard}' during the 
        activity: '{self.activity}' given how the hazard harms: '{self.how_it_harms}'
        and who/what the hazard harms: '{self.who_it_harms}'.
        2. Explain whether or not '{self.prevention}' reduces the likelihood of the hazard described in step 1.
        If so, it is a prevention measure.
        3. Assuming the hazard described in step 1 has already led to harm, explain whether or not '{self.prevention}'
        would reduce or remove the harm caused by the hazard described in step 1.
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'. 
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        prevention measure and a mitigation measure, answer 'both'.
        
        Use the following output format:
        Description: <your description>
        Prevention Explanation: <your prevention explanation>
        Mitigation Explanation: <your mitigation explanation>
        Answer: <your answer>'''
    
    def get_shortform_feedback(self):
        return ShortformFeedback(positive_feedback=f"Correct! '{self.prevention}' is a prevention measure for the hazard: '{self.hazard}'",
        negative_feedback=f"Incorrect. '{self.prevention}' is not a prevention measure for the hazard: '{self.hazard}'.")
    
    def get_longform_feedback(self, prompt_output):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Prevention Explanation', 'Mitigation')
    
class Mitigation(PromptInput):
    def __init__(self, mitigation, activity, hazard, how_it_harms, who_it_harms):
        super().__init__()
        self.mitigation = mitigation
        self.activity = activity
        self.hazard = hazard
        self.how_it_harms = how_it_harms
        self.who_it_harms = who_it_harms

        self.pattern_matching_method = 'check_string_for_prevention_mitigation_or_neither'
        self.correct_matched_pattern = 'mitigation'

    def get_question_title(self):
        return 'Mitigation'

    def get_question(self):
        return f'''Will the mitigation measure: '{self.mitigation}' reduce the severity of the
        'hazard': '{self.hazard}' occurring during the 'activity': {self.activity}, given
        given how the hazard harms: '{self.how_it_harms}' and who/what the hazard harms: '{self.who_it_harms}?'''

    def generate_prompt(self):

        example_of_correct_mitigation = '''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Ink spillage' during the
        activity: 'Fluids laboratory' given how the hazard harms: 'Serious eye damage'
        and who/what the hazard harms: 'Students'.
        2. Given the definition of a 'prevention measure': """an action which directly reduces the probability that the hazard occurs.""",
        explain whether 'Wash your eyes with clean water' is a prevention for hazard described in in step 1.
        3. Given the definition of a 'mitigation measure': """an action which directly reduces the harm caused by a hazard occurring
        or reduces the harm caused by the hazard after it has occurred.""",
        explain whether 'Wash your eyes with clean water' is a mitigation for hazard described in in step 1.
        4. If it is a prevention measure, answer prevention. If it is a migitation meausure, answer mitigation.
        If it is neither a prevention measure nor a mitigation measure, answer neither. If it is both a   
        prevention measure and a mitigation measure, answer both.

        Output: 
        Description: The hazard of 'Ink spillage' during the activity 'Fluids laboratory' can lead to serious eye damage to students.
        Prevention Explanation: 'Wash your eyes with clean water' is not a prevention measure; it is a response or first aid action. Prevention measures focus on reducing the probability that the hazard occurs in the first place, whereas washing the eyes is a reactive step taken after exposure.
        Mitigation Explanation: 'Wash your eyes with clean water' is a mitigation measure because it directly reduces the harm caused by the hazard after it has occurred by helping to minimize the potential damage to the eyes and providing initial care.
        Answer: Mitigation.'''

        example_of_prevention = '''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Wet hands' during the
        activity: 'Fluids laboratory' given how the hazard harms: 'Electric shock of students when touching electronics (pump power supply) with wet hands'
        and who/what the hazard harms: 'Students'.
        2. Given the definition of a 'prevention measure': """an action which directly reduces the probability that the hazard occurs.""",
        explain whether 'Students should make sure they touch electronics only with dry hands' is a prevention for hazard described in in step 1.
        3. Given the definition of a 'mitigation measure': """an action which directly reduces the harm caused by a hazard occurring
        or reduces the harm caused by the hazard after it has occurred.""",
        explain whether 'Students should make sure they touch electronics only with dry hands' is a mitigation for hazard described in in step 1.
        4. If it is a prevention measure, answer prevention. If it is a migitation meausure, answer mitigation.
        If it is neither a prevention measure nor a mitigation measure, answer neither. If it is both a   
        prevention measure and a mitigation measure, answer both.
        
        Output:
        Description: The hazard of 'Wet hands' during the activity 'Fluids laboratory' can lead to electric shock of students when touching electronics (pump power supply) with wet hands.
        Prevention Explanation: 'Students should make sure they touch electronics only with dry hands' is a prevention measure because it directly reduces the probability that the hazard of electric shock due to wet hands occurs.
        Mitigation Explanation: 'Students should make sure they touch electronics only with dry hands' is not a mitigation measure, as it does not directly reduce the harm caused by the hazard after it has occurred; instead, it focuses on preventing the occurrence of the hazard.
        Answer: Prevention'''

        example_of_incorrect_mitigation = '''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Exposure to toxic welding fumes' during the
        activity: 'Welding metal structures' given how the hazard harms: 'Inhaling welding fumes can lead to respiratory problems, lung damage, and long-term health issues.'
        and who/what the hazard harms: 'Welders and individuals in the vicinity of the welding area.'.    
        2. Given the definition of a 'prevention measure': """an action which directly reduces the probability that the hazard occurs.""",
        explain whether 'Using the welding equipment in an enclosed space without proper ventilation.' is a prevention for hazard described in in step 1.
        3. Given the definition of a 'mitigation measure': """an action which directly reduces the harm caused by a hazard occurring
        or reduces the harm caused by the hazard after it has occurred.""",
        explain whether 'Using the welding equipment in an enclosed space without proper ventilation.' is a mitigation for hazard described in in step 1.
        4. If it is a prevention measure, answer prevention. If it is a migitation meausure, answer mitigation.
        If it is neither a prevention measure nor a mitigation measure, answer neither. If it is both a   
        prevention measure and a mitigation measure, answer both.

        Output:
        Description: The hazard of 'Exposure to toxic welding fumes' during the activity 'Welding metal structures' can lead to inhaling welding fumes, resulting in respiratory problems, lung damage, and long-term health issues for welders and individuals in the vicinity of the welding area.
        Prevention Explanation: 'Using the welding equipment in an enclosed space without proper ventilation' is not a prevention measure; it exacerbates the hazard by creating a condition where exposure to toxic welding fumes is more likely to occur. 
        Mitigation Explanation: 'Using the welding equipment in an enclosed space without proper ventilation' is not a mitigation measure, as it does not directly reduce the harm caused by the hazard after it has occurred. Instead, it exacerbates the hazard by increasing the likelihood of exposure to toxic welding fumes. 
        Answer: neither'''

        return f'''
        {example_of_correct_mitigation}

        {example_of_prevention}

        {example_of_incorrect_mitigation}
        
        Follow these instructions:
        1. In one sentence, describe the hazard: '{self.hazard}' during the 
        activity: '{self.activity}' given how the hazard harms: '{self.how_it_harms}'
        and who/what the hazard harms: '{self.who_it_harms}'.
        2. Explain whether or not '{self.mitigation}' reduces the likelihood of the hazard described in step 1.
        If so, it is a prevention measure.
        3. Assuming the hazard described in step 1 has led to harm, explain whether or not '{self.mitigation}'
        would reduce or remove the harm caused by the hazard described in step 1.
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'. 
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        prevention measure and a mitigation measure, answer 'both'.
        
        Use the following output format:
        Description: <your description>
        Prevention Explanation: <your prevention explanation>
        Mitigation Explanation: <your mitigation explanation>
        Answer: <your answer>'''
    
    def get_shortform_feedback(self):
        return ShortformFeedback(positive_feedback=f"Correct! '{self.mitigation}' is a mitigation measure for the hazard: '{self.hazard}'.",
        negative_feedback=f"Incorrect. '{self.mitigation}' is not a mitigation measure for the hazard: '{self.hazard}'.")
    
    def get_longform_feedback(self, prompt_output):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Mitigation Explanation', 'Answer')