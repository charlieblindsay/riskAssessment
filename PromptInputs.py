try:
    from RegexPatternMatcher import RegexPatternMatcher
except:
    from .RegexPatternMatcher import RegexPatternMatcher

class PromptInput:
    def __init__(self):
        self.activity_definition = """an action or process that involves
        physical or mental effort."""

        self.hazard_definition = """a dangerous phenomenon, object, human activity or condition. 
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
        self.candidate_labels = [True, False]
        self.labels_indicating_correct_input = [True]

    def get_field_checked(self):
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

    def get_field_checked(self):
        return 'Activity'
    
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
        Overall Answer: <your answer>'''
    
    def get_shortform_feedback(self, feedback_type):
        if feedback_type == 'positive':
            return f"Correct! '{self.activity}' is an activity."
        if feedback_type == 'negative':
            return f"Incorrect. '{self.activity}' is not an activity."
    
    def get_longform_feedback(self, prompt_output):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Comparison', 'Overall Answer')

    def get_recommendation(self):
        return f'Enter an activity that aligns with the definition: {self.activity_definition}'

class HowItHarmsInContext(PromptInput):
    def __init__(self, how_it_harms, activity, hazard):
        super().__init__()
        self.how_it_harms = how_it_harms
        self.activity = activity
        self.hazard = hazard

    def get_field_checked(self):
        return 'Hazard & How It Harms'
    
    # TODO: Scope for adding a few shot example with an 'Add more detail' output.

    def generate_prompt_without_few_shot_examples(self):
        return f'''
        Follow these instructions:
        1. In one sentence, describe the hazard: '{self.hazard}' during the 
        activity: '{self.activity}'.
        2. In one sentence, explain whether or not '{self.how_it_harms}' is a way that this hazard causes harm. 
        3. If '{self.how_it_harms}' is a way that this hazard causes harm, answer True, else answer False.
        '''
    
    def generate_prompt(self):
        example_of_correct_how_it_harms = f'''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Electrocution' during the 
        activity: 'Fluids laboratory'.
        2. In one sentence, explain whether or not 'Electrocuted by mains voltage' is a way that this hazard causes harm. 
        3. If 'Electrocuted by mains voltage' is a way that this hazard causes harm, answer True, else answer False.

        Output:
        Description: It is argued that wet hands during a fluids laboratory can cause harm through electrocution.
        Explanation: As water is a conductor of electricity, touching electronics with wet hands can cause electrocution as
        the water provides a path for electrical current to flow through the body.
        Overall Answer: True
        '''

        example_of_incorrect_how_it_harms = f'''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Ink spillage' during the
        activity: 'Fluids laboratory'.
        2. In one sentence, explain whether or not 'Radiation exposure' is a way that this hazard causes harm.
        3. If 'Radiation exposure' is a way that this hazard causes harm, answer True, else answer False.

        Output:
        Description: It is argued that an ink spillage during a fluids laboratory can cause radiation exposure.
        Explanation: Radiation exposure is not a way that ink spillage during the fluids laboratory causes harm, 
        as the hazard primarily involves physical contamination rather than radiation.
        Overall Answer: False.
        '''
        return f'''
        {example_of_correct_how_it_harms}

        {example_of_incorrect_how_it_harms}

        {self.generate_prompt_without_few_shot_examples()}

        Use the following output format:
        Description: <your description>
        Explanation: <your Explanation>
        Overall Answer: <your answer>'''
    
    def get_shortform_feedback(self, feedback_type):
        if feedback_type == 'positive':
            return f"Correct! '{self.how_it_harms}' is a way that the hazard: '{self.hazard}' causes harm."
        if feedback_type == 'negative':
            return f"Incorrect. '{self.how_it_harms}' is not a way that the hazard: '{self.hazard}' causes harm."
    
    def get_longform_feedback(self, prompt_output):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Explanation', 'Overall Answer')

    def get_recommendation(self):
        return f'For the "How it harms" input, enter the type of injury or illness that the hazard event causes.'

class WhoItHarmsInContext(PromptInput):
    def __init__(self, who_it_harms, activity):
        super().__init__()
        self.who_it_harms = who_it_harms
        self.activity = activity

    def get_field_checked(self):
        return 'Who It Harms'
    
    def generate_prompt(self):
        
        example_of_correct_who_it_harms = f'''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the activity: 'Mucking out stable'.
        2. In one sentence, explain whether it is possible that a 'Stable hand' take part in this activity. 
        3. If it is possible, answer True, else answer False.

        Output:
        Description: "Mucking out a stable" involves the process of cleaning and removing waste, such as manure and soiled bedding, from a horse's stall or stable.
        Explanation: It is highly likely that a "Stable hand" would take part in this activity as it is a core responsibility associated with their role in maintaining the stable environment.
        Overall Answer: True
        '''

        example_of_incorrect_who_it_harms = f'''
        Example Input:
        1. In one sentence, describe the activity: 'Fluids laboratory'.
        2. In one sentence, explain whether it is possible that a 'Stable hand' take part in this activity.
        3. If it is possible, answer True, else answer False.

        Output:
        Description: A "Fluids laboratory" is a facility where controlled experiments and analyses are conducted to study the properties and behaviors of various fluids, including liquids and gases.
        Explanation: It is highly unlikely that a "Stable hand" would take part in this activity, as their expertise typically lies in the care and maintenance of horses within a stable environment rather than laboratory work with fluids.
        Overall Answer: False
        '''
        return f'''
        {example_of_correct_who_it_harms}

        {example_of_incorrect_who_it_harms}

        Follow these instructions:
        1. In one sentence, describe the activity: '{self.activity}'.
        2. In one sentence, explain whether it is possible that {self.who_it_harms} take part in this activity.
        3. If it is possible, answer True, else answer False.

        Your answer should be in the format:
        Description: <your description>
        Explanation: your_explanation
        Overall Answer: <your answer>'''
    
    def get_shortform_feedback(self, feedback_type):
        if feedback_type == 'positive':
            return f"Correct! '{self.who_it_harms}' could take part in the activity: '{self.activity}'."
        if feedback_type == 'negative':
            return f"Incorrect. '{self.who_it_harms}' could not take part in the activity: '{self.activity}'."

    def get_longform_feedback(self, prompt_output):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Explanation', 'Overall Answer')
    
    def get_recommendation(self):
        return f"For the 'Who it harms' field, please enter the individuals or group at risk of harm from the hazard"  

class ProtectiveClothing(PromptInput):
    def __init__(self, activity, hazard, who_it_harms, how_it_harms, control_measure):
        super().__init__()

        self.activity = activity
        self.hazard = hazard
        self.who_it_harms = who_it_harms
        self.how_it_harms = how_it_harms
        self.control_measure = control_measure

        self.labels_indicating_correct_input = [False]

    def get_field_checked(self):
        return 'Prevention and Mitigation'
    
    def generate_prompt_without_few_shot_examples(self):
        return f'''Follow these instructions:
        1. Description: In one sentence, describe the hazard: "{self.hazard}" during the
        activity: "{self.activity}" given how the hazard harms: "{self.how_it_harms}"
        and who the hazard harms: "{self.who_it_harms}".
        2. Protective Clothing Answer: If "{self.control_measure}" is an example of providing protective clothing, answer True, else answer False.
        3. Reduces Harm Explanation: Explain whether "{self.control_measure}" reduces the harm caused by the hazard.
        4. Reduces Harm Answer: If "{self.control_measure}" reduces the harm caused by the hazard, answer True, else answer False.
        5. If both "Protective Clothing Answer" and "Reduces Harm Answer" are True, answer True. Else answer False.'''
    
    def generate_prompt(self):
        example_of_correct_protective_clothing = '''
        Example Input:
        Follow these instructions:
        1. Description: In one sentence, describe the hazard: "Syringes with sharp needles" during the
        activity: "Fluids laboratory" given how the hazard harms: "Sharp needles can pierce the skin and cause bleeding"
        and who the hazard harms: "Students".
        2. Protective Clothing Answer: If "Wearing lab coat and PPE" is an example of providing protective clothing, answer True, else answer False.
        3. Reduces Harm Explanation: Explain whether "Wearing lab coat and PPE" reduces the harm caused by the hazard.
        4. Reduces Harm Answer: If "Wearing lab coat and PPE" reduces the harm caused by the hazard, answer True, else answer False.
        5. If both "Protective Clothing Answer" and "Reduces Harm Answer" are True, answer True. Else answer False.

        Output:
        Description: The hazard of "Syringes with sharp needles" during the activity "Fluids laboratory" can lead to sharp needles piercing the skin and causing bleeding to students.
        Protective Clothing Answer: True.
        Reduces Harm Explanation: "Wearing lab coat and PPE" provides a protective barrier between the needle and skin, thus reducing the harm caused by the hazard.
        Reduces Harm Answer: True. 
        Since "Protective Clothing Answer: True" and "Reduces Harm Answer: True", Overall Answer: True.
        '''

        return f'''

        {example_of_correct_protective_clothing}

        {self.generate_prompt_without_few_shot_examples()}

        Use the following output format:
        Description: <your description>
        Protective Clothing Answer: <your answer>
        Reduces Harm Explanation: <your explanation>
        Reduces Harm Answer: <your answer>
        Overall Answer: <your answer>'''
    
    def get_shortform_feedback(self, feedback_type):
        # Feedback is only given when a mitigation is written in prevention input.
        if feedback_type == 'positive':
            return f"Correct! '{self.control_measure}' is an example of a mitigation measure"
        if feedback_type == 'negative':
            return f"Incorrect. '{self.control_measure}' is not a prevention measure, but is actually a mitigation measure."
    
    def get_longform_feedback(self):
        return f"{self.control_measure} is an example of wearing protective clothing, which reduces the harm caused by the hazard event so is therefore a mitigation measure."
    
    # TODO: When you have hazard event input, can include below
    def get_recommendation(self):
        return f"""For the prevention field, enter a control measure which reduces the likelihood of the hazard event.
            Wearing protective clothing does not reduce the likelihood of the hazard event so it is not a prevention measure."""
    
class FirstAid(PromptInput):
    def __init__(self, activity, hazard, who_it_harms, how_it_harms, control_measure):
        super().__init__()

        self.activity = activity
        self.hazard = hazard
        self.who_it_harms = who_it_harms
        self.how_it_harms = how_it_harms
        self.control_measure = control_measure

    def get_field_checked(self):
        return 'Prevention and Mitigation'
    
    def generate_prompt_without_few_shot_examples(self):
        return f'''Follow these instructions:
        Follow these instructions:
        1. Describe "{self.control_measure}" in one sentence.
        2. Initial Medical Response Explanation: "Definition of initial medical response": "Immediate actions taken by individuals in the presence of a medical emergency or injury to provide basic care or initiate further assistance as needed.
        Given this "Definition of initial medical response", explain whether "{self.control_measure}" is an example of initial medical response.
        3. Initial Medical Response Answer: If "{self.control_measure}" is an example of an example of initial medical response, answer True, else answer False.
        4. Reduces Harm Explanation: Explain whether "{self.control_measure}" reduces the harm caused by the hazard: "{self.hazard}" given how it harms: "{self.how_it_harms}".
        5. Reduces Harm Answer: If "{self.control_measure}" reduces the harm, answer True, else answer False.
        6. If both "First Aid Answer" and "Reduces Harm Answer" are True, answer True, else answer False.'''
    
    def generate_prompt(self):
        example_of_correct_protective_clothing = '''
        Example Input:
        Follow these instructions:
        1. Description: Describe "Washing the eyes out with clean water" in one sentence.
        2. Initial Medical Response Explanation: "Definition of initial medical response": "Immediate actions taken by individuals in the presence of a medical emergency or injury to provide basic care or initiate further assistance as needed.
        Given this "Definition of initial medical response", explain whether ""Washing the eyes out with clean water"" is an example of initial medical response.
        3. Initial Medical Response Answer:  If "Washing the eyes out with clean water" is an example of initial medical response, answer True, else answer False.
        4. Reduces Harm Explanation: Explain whether "Washing the eyes out with clean water" reduces the harm caused by the hazard: "Ink spillage" given how it harms: "Serious eye damage".
        5. Reduces Harm Answer: If "Washing the eyes out with clean water" reduces the harm, answer True, else answer False.
        6. If both "First Aid Answer" and "Reduces Harm Answer" are True, answer True, else answer False.

        Output:
        Description: "Washing the eyes out with clean water" is the process of rinsing the eyes with clean water to remove any foreign bodies or chemicals that may have entered the eye.
        Intial Medical Response Explanation: "Washing the eyes out with clean water" is an example of basic care provided to an individual so is an example of "initial medical response".
        Initial Medical Response Answer: True.
        Reduces Harm Explanation: "Washing the eyes out with clean water" reduces the harm caused by the hazard as it helps to wash the ink out of the eyes and reduce eye damage.
        Reduces Harm Answer: True.
        Since "First Aid Answer: True" and "Reduces Harm Answer: True", Overall Answer: True.
        '''

        return f'''

        {example_of_correct_protective_clothing}

        {self.generate_prompt_without_few_shot_examples()}

        Use the following output format:
        Description: <your description>
        First Aid Answer: <your answer>
        Reduces Harm Explanation: <your explanation>
        Reduces Harm Answer: <your answer>
        Overall Answer: <your answer>'''
    
    def get_shortform_feedback(self, feedback_type):
        # Feedback is only given when a mitigation is written in prevention input.

        if feedback_type == 'positive':
            return f"Correct! '{self.control_measure}' is an example of a mitigation measure"
        if feedback_type == 'negative':
            return f"Incorrect. '{self.control_measure}' is not a prevention measure, but is actually a mitigation measure."
    
    def get_longform_feedback(self):
        return f"""{self.control_measure} is an example of a first aid measure, which reduces the harm caused by the hazard event after it has occurred; it is therefore a mitigation measure."""

    # TODO: When you have hazard event input, can include in feedback.
    def get_recommendation(self):
        return f"""For the prevention field, enter a control measure which reduces the likelihood of the hazard event.
        First aid is applied after the hazard event so does not reduce the likelihood of the hazard event occurring and is therefore not a prevention measure."""

class Prevention(PromptInput):
    def __init__(self, prevention, activity, hazard, how_it_harms, who_it_harms):
        super().__init__()
        self.prevention = prevention
        self.activity = activity
        self.hazard = hazard
        self.how_it_harms = how_it_harms
        self.who_it_harms = who_it_harms

        self.pattern_matching_method = 'check_string_for_prevention_mitigation_or_neither'
        self.candidate_labels = ['prevention', 'mitigation', 'neither', 'both']
        self.labels_indicating_correct_input = ['prevention', 'both']

    def get_field_checked(self):
        return 'Prevention'
    
    def generate_prompt_without_few_shot_examples(self):
        # return f'''Follow these instructions:
        # 1. In one sentence, describe the hazard: '{self.hazard}' during the 
        # activity: '{self.activity}' given how the hazard harms: '{self.how_it_harms}'
        # and who/what the hazard harms: '{self.who_it_harms}'.
        # 2. Explain whether or not '{self.prevention}' reduces the likelihood of the hazard described in step 1.
        # If so, it is a prevention measure.
        # 3. Assuming the hazard described in step 1 has already led to harm, explain whether or not '{self.prevention}'
        # would reduce or remove the harm caused by the hazard described in step 1.
        # If so, it is a mitigation measure.
        # 4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'. 
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        # prevention measure and a mitigation measure, answer 'both'.'''

        return f'''Follow these instructions:
        1. In one sentence, describe the hazard event: '{self.hazard}' during the 
        activity: '{self.activity}' given how the hazard harms: '{self.how_it_harms}'.
        2. Explain whether or not '{self.prevention}' reduces the likelihood that the hazard event occurs.
        If so, it is a prevention measure.
        3. If the hazard event occurs, explain whether or not '{self.prevention}' removes or reduces the chance of {self.how_it_harms}.
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'. 
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        prevention measure and a mitigation measure, answer 'Both'.'''
    
        # 2. In one sentence, explain why "{self.how_it_harms}" is a way that this hazard can cause harm. 
    
    def generate_prompt(self):


## Few shot exampls without chain of thought prompting
        # all_few_shot_examples = """
        # Follow these instructions:
        # 1. In one sentence, describe the hazard: 'Ink spillage' during the
        # activity: 'Fluids laboratory' given how the hazard harms: 'Serious eye damage'
        # and who the hazard harms: 'Students'.
        # 2. Describe the hazard event, which is the event that leads to harm.
        # 3. Explain whether or not 'First aid' reduces the likelihood that the hazard event occurs.
        # If so, it is a prevention measure.
        # 4. Assuming the hazard event occurs, explain whether or not 'First aid' removes or reduces the harm caused by the event.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        # prevention measure and a mitigation measure, answer 'Both'.

        # Output: 
        # Hazard Description: The hazard of 'Ink spillage' during the activity 'Fluids laboratory' can lead to serious eye damage to students.
        # Hazard Event Description: Ink being spilled onto a student's face.
        # Prevention Explanation: 'First aid' will not reduce the likelihood of ink being spilled on the student's face; it is therefore not a prevention measure.
        # Mitigation Explanation: As it reduces the harm caused by the hazard event, it is therefore a mitigation measure.
        # Answer: Mitigation.

        # Follow these instructions:
        # 1. In one sentence, describe the hazard: 'Water being spilt on the floor' during the
        # activity: 'Fluids laboratory' given how the hazard harms: 'Injuries caused by possible slipping on wet floor'
        # and who the hazard harms: 'Students'.
        # 2. Describe the hazard event, which is the event that leads to harm.
        # 3. Explain whether or not 'Do not move the water tank when it is full' reduces the likelihood that the hazard event occurs.
        # If so, it is a prevention measure.
        # 4. Assuming the hazard event occurs, explain whether or not 'Do not move the water tank when it is full' removes or reduces the harm caused by the event.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        # prevention measure and a mitigation measure, answer 'Both'.

        # Hazard Description: The hazard of 'Water being spilt on the floor' during the activity 'Fluids laboratory' can lead to injuries caused by possible slipping on a wet floor to students.
        # Hazard Event Description: Water is accidentally spilled on the floor.
        # Prevention Explanation: As it reduces the likelihood of the hazard event, it is a prevention measure.
        # Mitigation Explanation: As it does not reduce the harm caused by the hazard event, it is not a mitigation measure.
        # Answer: Prevention.

        # Follow these instructions:
        # 1. In one sentence, describe the hazard: 'Loud noise' during the
        # activity: 'Using a trombone as a demonstration for a TPS presentation' given how the hazard harms: 'Loud noise from instrument can cause hearing damage.'
        # and who the hazard harms: 'Everyone present'.
        # 2. Describe the hazard event, which is the event that leads to harm.
        # 3. Explain whether or not 'Keep a space between the player and audience' reduces the likelihood that the hazard event occurs.
        # If so, it is a prevention measure.
        # 4. Assuming the hazard event occurs, explain whether or not 'Keep a space between the player and audience' removes or reduces the harm caused by the event.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        # prevention measure and a mitigation measure, answer 'Both'.

        # Hazard Description: The hazard of 'Loud noise' during the activity 'Using a trombone as a demonstration for a TPS presentation' can cause hearing damage to everyone present.
        # Hazard Event Description: The trombone player plays the instrument at a high volume, producing a loud noise.
        # Prevention Explanation: As it does not reduce the likelihood of the hazard event, it is not a prevention measure.
        # Mitigation Explanation: As it reduces the harm caused by the hazard event, it is a mitigation measure.
        # Answer: Mitigation.

        # Follow these instructions:
        # 1. In one sentence, describe the hazard: 'Syringes with sharp needles' during the
        # activity: 'Fluids laboratory' given how the hazard harms: 'Sharp needles can pierce the skin and cause bleeding'
        # and who the hazard harms: 'Students'.
        # 2. Describe the hazard event, which is the event that leads to harm.
        # 3. Explain whether or not 'Wear lab coat and PPE' reduces the likelihood that the hazard event occurs.   
        # If so, it is a prevention measure.
        # 4. Assuming the hazard event occurs, explain whether or not 'Wear lab coat and PPE' removes or reduces the harm caused by the event.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        # prevention measure and a mitigation measure, answer 'Both'.

        # Hazard Description: The hazard of 'Syringes with sharp needles' during the activity 'Fluids laboratory' can lead to sharp needles piercing the skin and causing bleeding to students.
        # Hazard Event Description: A sharp syringe needle is directed towards an student.
        # Prevention Explanation: As it does not reduce the likelihood of the hazard event, it is therefore not a prevention measure.
        # Mitigation Explanation: As it reduces the harm caused by the hazard event, it is a mitigation measure.
        # Answer: Mitigation.

        # Follow these instructions:
        # 1. In one sentence, describe the hazard: 'Water from instrument' during the
        # activity: 'Using a trombone as a demonstration for a TPS presentation' given how the hazard harms: 'Condensation formed in instrument could spread germs if released'
        # and who the hazard harms: 'Audience'.
        # 2. Describe the hazard event, which is the event that leads to harm.
        # 3. Explain whether or not 'Keep a space between the player and audience' reduces the likelihood that the hazard event occurs.
        # If so, it is a prevention measure.
        # 4. Assuming the hazard event occurs, explain whether or not 'Keep a space between the player and audience' removes or reduces the harm caused by the event.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        # prevention measure and a mitigation measure, answer 'Both'.
        
        # Hazard Description: The hazard of 'Water from instrument' during the activity 'Using a trombone as a demonstration for a TPS presentation' can lead to the spread of germs to the audience if condensation formed in the instrument is released.
        # Hazard Event Description: Water from the trombone condenses and is released into the air.
        # Prevention Explanation: As it does not reduce the likelihood of the hazard event, it is not a prevention measure.
        # Mitigation Explanation: As it reduces the harm caused by the hazard event, it is a mitigation measure.
        # Answer: Mitigation.

        # Follow these
        # Input: instructions:
        # 1. In one sentence, describe the hazard: 'Students catch germs from the water' during the
        # activity: 'Using a trombone as a demonstration for a TPS presentation' given how the hazard harms: 'Illnesses from germs'.
        # 3. Explain whether or not 'Keep a space between the player and audience' reduces the likelihood that the hazard event occurs.
        # If so, it is a prevention measure.
        # 4. Assuming the hazard event occurs, explain whether or not 'Keep a space between the player and audience' removes or reduces the chance of 'Illnesses from germs'.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        # prevention measure and a mitigation measure, answer 'Both'.
        
        # Hazard Description: The hazard of 'Students catch germs from the water' during the activity 'Using a trombone as a demonstration for a TPS presentation' can lead to the spread of germs to the audience if condensation formed in the instrument is released.
        # Hazard Event Description: Water from the trombone condenses and is released into the air.
        # Prevention Explanation: 'Keeping a space between the player and the audience' does not reduce the likelihood of water condensing in the instrument or being released; as it does not reduce the likelihood of the hazard event, it is not a prevention measure.
        # Mitigation Explanation: If water from the instrument is released, 'keeping a space between the player and the audience' will reduce the likelihood of germs reaching the audience; as it reduces the harm caused by the hazard event, it is a mitigation measure.
        # Answer: Mitigation.

        # Follow these
        # Input: instructions:
        # 1. In one sentence, describe the hazard: 'Sharp syringe needles poked into student' during the
        # activity: 'Fluids laboratory' given how the hazard harms: 'Sharp needles can pierce the skin and cause bleeding'.
        # 3. Explain whether or not 'Wear lab coat and PPE' reduces the likelihood that the hazard event occurs.   
        # If so, it is a prevention measure.
        # 4. Assuming the hazard event occurs, explain whether or not 'Wear lab coat and PPE' removes or reduces the chance of 'Sharp needles can pierce the skin and cause bleeding'.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        # prevention measure and a mitigation measure, answer 'Both'.

        # Hazard Description: The hazard of 'Syringes with sharp needles' during the activity 'Fluids laboratory' can lead to sharp needles piercing the skin and causing bleeding to students.
        # Hazard Event Description: A sharp syringe needle is directed towards an student.
        # Prevention Explanation: 'Wearing a lab coat and personal protective equipment (PPE)' does not reduce the likelihood of a student directing a syringe needle towards another student; as it does not reduce the likelihood of the hazard event, it is therefore not a prevention measure.
        # Mitigation Explanation: If a sharp syringe needle is directed towards a student, 'wearing a lab coat and PPE' will reduce the harm caused by the sharp needle as it is unlikely to pierce through the lab coat and PPE; as it reduces the harm caused by the hazard event, it is a mitigation measure.
        # Answer: Mitigation.

        # Follow these
        # Input: instructions:
        # 1. In one sentence, describe the hazard: 'Loud noise' during the
        # activity: 'Using a trombone as a demonstration for a TPS presentation' given how the hazard harms: 'Hearing damage.'
        # 3. Explain whether or not 'Keep a space between the player and audience' reduces the likelihood that the hazard event occurs.
        # If so, it is a prevention measure.
        # 4. Assuming the hazard event occurs, explain whether or not 'Keep a space between the player and audience' removes or reduces the chance of 'Hearing damage'.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        # prevention measure and a mitigation measure, answer 'Both'.

        # Hazard Description: The hazard of 'Loud noise' during the activity 'Using a trombone as a demonstration for a TPS presentation' can cause hearing damage to everyone present.
        # Hazard Event Description: The trombone player plays the instrument at a high volume, producing a loud noise.
        # Prevention Explanation: 'Keeping a space between the player and the audience' does not reduce the likelihood of the trombone producing a loud noise. As it does not reduce the likelihood of the hazard event, it is not a prevention measure.
        # Mitigation Explanation: If the hazard event occurs and the trombone produces a loud noise, 'keeping a space between the player and the audience' will reduce the noise heard by the audience, hence reducing the severity of the hearing damage caused by the loud noise; as it reduces the harm caused by the hazard event, it is a mitigation measure.
        # Answer: Mitigation.
        
        # """

        # TODO: There are 4 mitigations, and 1 prevention!

        all_few_shot_examples = """
        Input:
        Follow these instructions:
        1. In one sentence, describe the hazard event: 'Ink spillage on students face' during the
        activity: 'Fluids laboratory' given how the hazard event causes harm: 'Serious eye damage'.
        3. Explain whether or not 'First aid' reduces the likelihood that the hazard event occurs.
        If so, it is a prevention measure.
        4. Assuming the hazard event occurs, explain whether or not 'First aid' removes or reduces the chance of 'Serious eye damage'.
        If so, it is a mitigation measure.
        5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        prevention measure and a mitigation measure, answer 'Both'.

        Output: 
        Hazard Description: The hazard event of 'Ink spillage on student's face' during the activity 'Fluids laboratory' can lead to serious eye damage to students.
        Prevention Explanation: 'First aid' is a reactive measure applied after the hazard event of 'Ink spillage on student's face'; it therefore does not reduce the likelihood of the hazard event and is not a prevention measure.
        Mitigation Explanation: If ink has been spilled onto a student's face, 'first aid' will help to wash the ink out of the eyes and reduce eye damage after the hazard event has occurred; as it reduces the harm caused by the hazard event, it is therefore a mitigation measure.
        Answer: Mitigation.

        Follow these
        Input: instructions:
        1. In one sentence, describe the hazard event: 'Water being spilt on the floor causing students to slip' during the
        activity: 'Fluids laboratory' given how the hazard harms: 'Impact injury'.
        3. Explain whether or not 'Do not move the water tank when it is full' reduces the likelihood that the hazard event occurs.
        If so, it is a prevention measure.
        4. Assuming the hazard event occurs, explain whether or not 'Do not move the water tank when it is full' removes or reduces the chance of 'Impact injury'.
        If so, it is a mitigation measure.
        5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        prevention measure and a mitigation measure, answer 'Both'.

        Hazard Description: The hazard of 'Water being spilt on the floor causing students to slip' during the activity 'Fluids laboratory' can lead to impact injuries.
        Prevention Explanation: 'Keeping the water tank stationary when it's full' means water cannot be spilled on to the floor by moving the water tank; no water on the floor reduces the likelihood of the student slipping; since it reduces the likelihood of the hazard event, it is a prevention measure.
        Mitigation Explanation: If water has been spilled on the floor, 'not moving the water tank when it is full' does not remove or reduce the harm caused by the hazard event, as the water is already spilled to pose a slipping hazard; as it does not reduce the harm caused by the hazard event, it is not a mitigation measure.
        Answer: Prevention.

        Follow these instructions:
        1. In one sentence, describe the hazard event: 'Cut Zip tie flies and hits audience member' during the
        activity: 'Using a spring contraption as a demonstration for a TPS presentation' given how the hazard harms: 'Impact injury.'.
        2. Explain whether or not 'Keep hand around zip tie when cutting to stop it from flying' reduces the likelihood that the hazard event occurs.
        If so, it is a prevention measure.
        3. If the hazard event occurs, explain whether or not 'Keep hand around zip tie when cutting to stop it from flying' removes or reduces the chance of Impact injury..
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a
        prevention measure and a mitigation measure, answer 'Both'.

        Hazard Description: The hazard event of 'Cut Zip tie flies and hits audience member' during the activity 'Using a spring contraption as a demonstration for a TPS presentation' can lead to impact injuries.
        Prevention Explanation: 'Keeping hand around zip tie when cutting to stop it from flying' will stop the zip tie from flying and therefore stop the hazard event from occurring. Therefore, the likelihood of the hazard event occurring has been reduced to zero; since the likelihood has been reduced, it is therefore a prevention measure.
        Mitigation Explanation: If the hazard event occurs and the zip tie flies and hits an audience member, 'keeping hand around zip tie when cutting to stop it from flying' does not remove or reduce the impact injury caused by the hazard event, as the zip tie has already flown and caused harm; it is therefore not a mitigation measure.
        Answer: Prevention.
        """

        return f'''
        {all_few_shot_examples}

        {self.generate_prompt_without_few_shot_examples()}

        Use the following output format:
        Hazard Description: <your hazard description>
        Hazard Event Description: <your hazard event description>
        Prevention Explanation: <your prevention explanation>
        Mitigation Explanation: <your mitigation explanation>
        Answer: <your answer>'''
    
        #     # How it Harms Explanation: <your how it harms explanation>

        # return f'''
        # {self.generate_prompt_without_few_shot_examples()}

        # Use the following output format:
        # Hazard Description: <your hazard description>
        # Hazard Event Description: <your hazard event description>
        # Prevention Explanation: <your prevention explanation>
        # Mitigation Explanation: <your mitigation explanation>
        # Answer: <your answer>'''
    
        # How it Harms Explanation: <your how it harms explanation>
    
    def get_shortform_feedback(self, feedback_type):
        if feedback_type == 'positive':
            return f"Correct! '{self.prevention}' is a prevention measure for the hazard: '{self.hazard}'"
        if feedback_type == 'neither':
            return f"Incorrect. '{self.prevention}' is not a prevention measure for the hazard: '{self.hazard}'."
        if feedback_type == 'misclassification':
            return f"Incorrect. '{self.prevention}' is actually a mitigation measure for the hazard: '{self.hazard}'."
    
    def get_longform_feedback(self, prompt_output, pattern_to_search_for='Prevention Explanation', lookahead_assertion='Mitigation'):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, pattern_to_search_for, lookahead_assertion)

    # TODO: When you have hazard event input, can include in feedback.
    def get_recommendation(self, recommendation_type):
        if recommendation_type == 'neither':
            return "For the prevention field, enter a control measure which reduces the likelihood of the hazard event."
        
        if recommendation_type == 'misclassification':
            return f"""A mitigation measure reduces the harm caused by the hazard event either while the hazard event is occurring or after it has occurred.
            On the other hand, a prevention measure reduces the likelihood of the hazard event occurring in the first place.
            Please use the above definitions to ammend your prevention input."""
    
class Mitigation(PromptInput):
    def __init__(self, mitigation, activity, hazard, how_it_harms, who_it_harms):
        super().__init__()
        self.mitigation = mitigation
        self.activity = activity
        self.hazard = hazard
        self.how_it_harms = how_it_harms
        self.who_it_harms = who_it_harms

        self.pattern_matching_method = 'check_string_for_prevention_mitigation_or_neither'
        self.candidate_labels = ['prevention', 'mitigation', 'neither', 'both']
        self.labels_indicating_correct_input = ['mitigation', 'both']

    def get_field_checked(self):
        return 'Mitigation'

    def get_question(self):
        return f'''Will the mitigation measure: '{self.mitigation}' reduce the severity of the
        'hazard': '{self.hazard}' occurring during the 'activity': {self.activity}, given
        given how the hazard harms: '{self.how_it_harms}' and who/what the hazard harms: '{self.who_it_harms}?'''
    
    def generate_prompt_without_few_shot_examples(self):
        # return f'''Follow these instructions:
        # 1. In one sentence, describe the hazard: '{self.hazard}' during the 
        # activity: '{self.activity}' given how the hazard harms: '{self.how_it_harms}'
        # and who the hazard harms: '{self.who_it_harms}'.
        # 2. In one sentence, explain why "{self.how_it_harms}" is a way that this hazard can cause harm. 
        # 3. Explain whether or not '{self.mitigation}' reduces the likelihood that the hazard causes harm.
        # If so, it is a prevention measure.
        # 4. Assuming the hazard described above does harm someone, explain whether or not '{self.mitigation}' reduces the harm.
        # If so, it is a mitigation measure.
        # 5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'. 
        # If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        # prevention measure and a mitigation measure, answer 'Both'.'''
        return f'''Follow these instructions:
        1. In one sentence, describe the hazard: '{self.hazard}' during the 
        activity: '{self.activity}' given how the hazard harms: '{self.how_it_harms}'
        and who the hazard harms: '{self.who_it_harms}'.
        2. In one sentence, explain why {self.how_it_harms} is a way that this hazard can cause harm.
        3. Explain whether or not '{self.mitigation}' reduces the likelihood that the hazard event occurs.
        If so, it is a prevention measure.
        4. Assuming the hazard event occurs, explain whether or not '{self.mitigation}' removes or reduces the harm caused by the event.
        If so, it is a mitigation measure.
        5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'. 
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a 
        prevention measure and a mitigation measure, answer 'Both'.'''
    
    def generate_prompt(self):

        example_of_correct_mitigation_where_mitigation_reduces_harm_after_hazard_event_has_occurred = f'''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Ink spillage' during the
        activity: 'Fluids laboratory' given how the hazard harms: 'Serious eye damage'
        and who the hazard harms: 'Students'.
        2. In one sentence, explain why "Serious eye damage" is a way that this hazard can cause harm.
        3. Explain whether or not 'First aid' reduces the likelihood that the hazard causes harm.
        If so, it is a prevention measure.
        4. Assuming the hazard described above does harm someone, explain whether or not 'First aid' reduces the harm. 
        If so, it is a mitigation measure.
        5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        prevention measure and a mitigation measure, answer 'Both'.

        Output: 
        Description: The hazard of 'Ink spillage' during the activity 'Fluids laboratory' can lead to serious eye damage to students.
        How it Harms Explanation: 'Serious eye damage' is a way that this hazard causes harm because if ink comes into contact with the eyes, it can cause serious damage.
        Prevention Explanation: First aid will not reduce the likelihood of an ink spillage causing harm and it is a reactive step taken after the ink spillage; it therefore does not reduce the likelihood that the hazard causes harm and is not a prevention measure.
        Mitigation Explanation: If an ink spillage has led to serious eye damagen, first aid will help to wash the ink out of the eyes and reduce eye damage; as it reduces the harm, it is therefore a mitigation measure.
        Answer: Mitigation.'''

        example_of_mitigation_which_reduces_harm_when_hazard_event_is_occurring = '''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Horse kicks out' during the
        activity: 'Mucking out a horse' given how the hazard harms: 'Impact injury'
        and who the hazard harms: 'Horse rider'.
        2. In one sentence, explain why "Impact injury" is a way that this hazard can cause harm.
        3. Explain whether or not 'Wear a helmet and body protector' reduces the likelihood that the hazard causes harm.
        If so, it is a prevention measure.
        4. Assuming the hazard described above does harm someone, explain whether or not 'Wear a helmet and body protector' reduces the harm.
        If so, it is a mitigation measure.
        5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        prevention measure and a mitigation measure, answer 'Both'.
        
        Output:
        Description: The hazard of 'Horse kicks out' during the activity 'Mucking out a horse' can lead to impact injury to the horse rider.
        How it Harms Explanation: When a horse kicks out during mucking out, it can cause harm through impact injury, as the force of the kick can lead to bruises, fractures, or other injuries.
        Prevention Explanation: Wearing a helmet and body protector does not reduce the likelihood that the horse will kick and is therefore not a prevention measure.
        Mitigation Explanation: If a horse kicks the horse rider, wearing a helmet and body protector provides a protective barrier between the horse's kick and the person, hence reducing the impact injury caused by the horse's kick; as it reduces the harm, it is therefore a mitigation measure.
        Answer: Mitigation.
        '''
        example_of_prevention = f'''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Tripping over personal belongings' during the
        activity: 'Fluids laboratory' given how the hazard harms: 'Tripping can cause physical harm.'
        and who the hazard harms: 'Students'.
        2. In one sentence, explain why "Tripping can cause physical harm." is a way that this hazard can cause harm.
        3. Explain whether or not 'Take care when walking around' reduces the likelihood that the hazard causes harm.
        If so, it is a prevention measure.
        4. Assuming the hazard described above does harm someone, explain whether or not 'Take care when walking around' reduces the harm.
        If so, it is a mitigation measure.
        5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        prevention measure and a mitigation measure, answer 'Both'.

        Output:
        Description: The hazard of 'Tripping over personal belongings' during the activity 'Fluids laboratory' can lead to physical harm to students.
        How it Harms Explanation: Tripping can cause physical harm because it can result in falls, which can lead to injuries such as bruises, sprains, or fractures.
        Prevention Explanation: 'Take care when walking around' encourages students to be cautions and aware of their surroundings, making it less likely they will trip so it is a prevention measure.
        Mitigation Explanation: If a student has tripped over personal belongings, whether or not they were taking care when walking around will not affect how much harm the trip; as it does not reduce harm, it is therefore not a mitigation measure.
        Answer: Prevention.'''

        return f'''
        {example_of_correct_mitigation_where_mitigation_reduces_harm_after_hazard_event_has_occurred}

        {example_of_mitigation_which_reduces_harm_when_hazard_event_is_occurring}

        {example_of_prevention}
        
        {self.generate_prompt_without_few_shot_examples()}

        Use the following output format:
        Description: <your description>
        How it Harms Explanation: <your how it harms explanation>
        Prevention Explanation: <your prevention explanation>
        Mitigation Explanation: <your mitigation explanation>
        Answer: <your answer>'''
    
    def get_shortform_feedback(self, feedback_type):
        if feedback_type == 'positive':
            return f"Correct! '{self.mitigation}' is a mitigation measure for the hazard: '{self.hazard}'."
        if feedback_type == 'neither':
            return f"Incorrect. '{self.mitigation}' is not a mitigation measure for the hazard: '{self.hazard}'."
        if feedback_type == 'misclassification':
            return f"Incorrect. '{self.mitigation}' is actually a prevention measure for the hazard: '{self.hazard}'."
    
    def get_longform_feedback(self, prompt_output, pattern_to_search_for='Mitigation Explanation', lookahead_assertion='Answer'):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, pattern_to_search_for, lookahead_assertion)
    
    # TODO: When you have hazard event input, can include in feedback.
    def get_recommendation(self, recommendation_type):
        if recommendation_type == 'neither':
            return "For the mitigation field, enter a control measure which reduces the harm caused by the hazard event either while the hazard event is occurring or after it has occurred."
        
        if recommendation_type == 'misclassification':
            return f"""A prevention measure reduces the likelihood of the hazard event occurring in the first place.
            On the other hand, a mitigation measure reduces the harm caused by the hazard event while it is happening or after it has occurred.
            Please use the above definitions to ammend your mitigation input."""