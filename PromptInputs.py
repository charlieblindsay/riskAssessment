try:
    from RegexPatternMatcher import RegexPatternMatcher
except:
    from .RegexPatternMatcher import RegexPatternMatcher

class PromptInput:
    def __init__(self):
        self.activity_definition = """an action or process that involves physical or mental effort."""

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

class InputFieldClassification(PromptInput):
    def __init__(self, input, field_name):
        super().__init__()
        self.input = input
        self.field_name = field_name

        self.pattern_matching_method = 'check_string_for_type_of_input_field'

        self.candidate_labels = ['activity', 'hazard', 'event that leads to harm', 'harm caused', 'who', 'control measure']

    def get_correct_labels(self):
        if self.field_name == 'Activity':
            return ['activity', 'hazard', 'event that leads to harm']
        if self.field_name == 'Hazard':
            return ['activity', 'hazard', 'event that leads to harm', 'harm caused', 'who']
        if self.field_name == 'How it harms':
            return ['hazard', 'event that leads to harm', 'harm caused'] # Classifying these 3 comes later
        if self.field_name == 'Who is harmed by this event':
            return ['who', 'hazard'] # E.g. "Reckless drivers" would be suitable for hazard or who it harms.
        if self.field_name == 'Prevention' or self.field_name == 'Mitigation':
            return ['control measure']
        
    def generate_prompt(self):
        return f'''
        Classify the following as either an {', '.join(self.candidate_labels)}:

        Input: Playing at a Playground
        Answer: Activity

        Input: Broken equipment
        Answer: Hazard
        
        Input: Child falling off monkey bars
        Answer: Event that leads to harm

        Input: Fractured arm
        Answer: Harm caused

        Input: Horse rider
        Answer: Who

        Input: Wearing a helmet
        Answer: Control Measure

        Classify the following as either an {', '.join(self.candidate_labels)}:
        Input: {self.input}

        Use the following format:
        Answer: <your answer>'''
        
    # def generate_prompt(self):
    #     return f'''
    #     Follow these instructions:
    #     1. In one sentence, describe the following: <input>
    #     2. Classify the following as either an {', '.join(self.candidate_labels)}:

    #     Input: Playing at a Playground
    #     1. Description: "Playing at a playground" involves children engaging in physical activities.
    #     2. Answer: Activity

    #     Input: Broken equipment
    #     1. Description: "Broken equipment" is a dangerous phenomenon that can cause injury or other health impacts.
    #     2. Answer: Hazard
        
    #     Input: Child falling off monkey bars
    #     1. Description: "Child falling off monkey bars" involves a child losing their grip on the monkey bars and falls to the ground below.
    #     2. Answer: Event that leads to harm

    #     Input: Fractured arm
    #     1. Description: "Fractured arm" is a type of injury.
    #     2. Answer: Harm caused

    #     Input: Horse rider
    #     1. Description: "Horse rider" is an individual who rides a horse.
    #     2. Answer: Who

    #     Input: Wearing a helmet
    #     1. Description: "Wearing a helmet" involves placing a helmet on one's head to provide protection in the event of impact or accident.
    #     2. Answer: Control Measure

    #     Classify the following as either an {', '.join(self.candidate_labels)}:
    #     Input: {self.input}

    #     Use the following format:
    #     1. Description: <your description>
    #     2. Answer: <your answer>'''
    
    def get_input_field_from_pattern_matched(self, pattern_matched):
        if pattern_matched == 'who':
            return 'Who is harmed by this event'
        if pattern_matched == 'harm caused':
            return 'How it harms'
        if pattern_matched == 'control measure':
            return 'Prevention or Mitigation'
        else:
            return pattern_matched.capitalize()

    def get_shortform_feedback(self, pattern_matched):
        return f"""Incorrect! '{self.input}' is not a correct input for the field: '{self.field_name}'. It looks more like an example of the field: '{self.get_input_field_from_pattern_matched(pattern_matched=pattern_matched)}'."""

class Activity(PromptInput):
    def __init__(self, activity: str):
        super().__init__()
        self.activity = activity

    def get_field_checked(self):
        return 'Activity'
    
        #     Input: Playing at a Playground
        # Answer: True

        # Input: Broken equipment
        # Answer: False

        # Input: Child falling off monkey bars
        # Answer: True

        # Input: Golf
        # Answer: True

        # Input: Chess
        # Answer: True

        # Input: Fluids laboratory
        # Answer: True
    
    def generate_prompt(self):
        return f'''
        If the input is an example of an activity, answer True, else answer False.

        If "{self.activity}" is an example of an activity, answer True, else answer False.

        Use the following output format:
        Overall Answer: <your answer>'''
    
    def get_shortform_feedback(self, feedback_type):
        if feedback_type == 'positive':
            return f"Correct! '{self.activity}' is an activity."
        if feedback_type == 'negative':
            return f"Incorrect. '{self.activity}' is not an activity."
    
    def get_longform_feedback(self, prompt_output=''):
        regex_pattern_matcher = RegexPatternMatcher()
        return ''

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
    
    def get_longform_feedback(self, prompt_output=''):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, 'Explanation')

    def get_recommendation(self):
        return f'For the "How it harms" input, enter how the hazard leads to harm, and the specific harm caused, e.g. an injury or illness.'

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

    def get_longform_feedback(self, prompt_output=''):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, 'Explanation')
    
    def get_recommendation(self):
        return f"For the 'Who it harms' field, please enter the individuals or group at risk of harm from the hazard"  
    
# class Event(PromptInput):
#     def __init__(self, hazard_event):
#         super().__init__()
#         self.hazard_event = hazard_event
#         self.pattern_matching_method = 'check_string_for_true_or_false_with_no_overall_answer'
    
#     def get_field_checked(self):
#         return 'Hazard Event'
    
#     def generate_prompt(self):
#         return f'''
#         Follow these instructions:
#         For the following input, if it is an event, answer True, else answer False.

#         Input: Unsecured scaffolding
#         Answer: False

#         Input: A worker falling from a height
#         Answer: True

#         Input: Impact injury
#         Answer: False

#         Input: Operator caught in machinery
#         Answer: True

#         Input: Construction workers
#         Answer: False

#         Input: Swimmer drowning
#         Answer: True

#         For the following input, if it is an event, answer True, else answer False.
#         Input: {self.hazard_event}
        
#         Use the following output format:
#         Answer: <your answer>'''

# class GetHarmCaused(PromptInput):
#     def __init__(self, how_it_harms):
#         super().__init__()
#         self.how_it_harms = how_it_harms
    
#     def get_field_checked(self):
#         return 'How it harms'
    
#     def generate_prompt(self):
#         return f'''

#         Follow these instructions:
#         1. An injury is defined as physical damage inflicted on the body.
#         Explain whether the input aligns with the definition of an injury.
#         2. If the input contains an example of an injury, give the injury, else write Injury: False.
#         3. An illness is defined as a disease or sickness affecting the body or mind.
#         Explain whether the input aligns with the definition of an illness.
#         4. If the input contains an example of an illness, give the illness, else write Illness: False.

#         Input: A worker falling from a height
#         Injury Explanation: A worker falling from a height is a description of the event which leads to harm, not the harm itself.
#         Injury Answer: False
#         Illness Explanation: A worker falling from a height is not a disease or sickness affecting the body or mind.
#         Illness Answer: False
#         Since both Injury Answer and Illness Answer are False, Overall Answer: neither

#         Input: It results in a burn.
#         Injury Explanation: A burn is an example of physical damage inflicted on the body.
#         Injury Answer: Burn
#         Illness Explanation: A burn is not a disease or sickness affecting the body or mind.
#         Illness Answer: False
#         Since Injury Answer is not False, Overall Answer: Burn

#         Input: Prolonged exposure to high temperatures without proper precautions
#         Injury Explanation: Prolonged exposure to high temperatures without proper precautions is a description of the event which leads to an harm, not the harm itself.
#         Injury Answer: False
#         Illness Explanation: Prolonged exposure to high temperatures without proper precautions is a description of the event which leads to an harm, not the harm itself.
#         Illness Answer: False
#         Since both Injury Answer and Illness Answer are False, Overall Answer: neither

#         Input: Foodborne illnesses due to improper handling of food items
#         Injury Explanation: Foodborne illnesses are a type of disease or sickness affecting the body or mind, so are an illness.
#         Injury Answer: False
#         Illness Explanation: Foodborne illnesses are a type of disease or sickness affecting the body or mind.
#         Illness Answer: Foodborne illnesses
#         Since Illness Answer is not False, Overall Answer: Foodborne illnesses

#         1. An injury is defined as damage inflicted on the body.
#         Explain whether "{self.input}" aligns with the definition of an injury.
#         2. If the input contains an example of an injury, give the injury, else write Injury: False.
#         3. An illness is defined as a disease or sickness affecting the body or mind.
#         Explain whether "{self.input}" aligns with the definition of an illness.
#         4. If the input contains an example of an illness, give the illness, else write Illness: False.
#         5. If both the Injury Answer and Illness Answer are False, Overall Answer: neither. If the Injury Answer is not False, Overall Answer: <your injury answer>. If the Illness Answer is not False, Overall Answer: <your illness answer>.

#         Use the following output format:
#         Injury Explanation: <your injury explanation>
#         Injury: <your injury>
#         Illness Explanation: <your illness explanation>
#         Illness: <your illness>'''

class Injury(PromptInput):
    def __init__(self, input):
        super().__init__()
        self.input = input
        self.pattern_matching_method = 'extract_injury'

        self.candidate_labels = ['injury', 'illness', 'neither']
    
    def get_field_checked(self):
        return 'How it harms'
    
    def generate_prompt(self):
        return f'''

        Follow these instructions:
        1. An injury is defined as physical damage inflicted on the body.
        In one sentence, explain whether the input contains a phrase which aligns with the definition of an injury.
        2. If the input contains an example of an injury, give the injury, else write Injury: False.

        Input: An impact injury is caused
        Explanation: An impact injury is a type of physical damage inflicted on the body.
        Injury: Impact injury

        Input: A worker falling from a height
        Explanation: A worker falling from a height is a description of the event which leads to harm, not the harm itself.
        Injury: False

        Input: It results in a burn.
        Explanation: A burn is an example of physical damage inflicted on the body.
        Injury: Burn

        Input: Prolonged exposure to high temperatures without proper precautions
        Explanation: Prolonged exposure to high temperatures without proper precautions is a description of the event which leads to an harm, not the harm itself.
        Injury: False

        Input: Foodborne illnesses due to improper handling of food items
        Explanation: Foodborne illnesses are a type of disease or sickness affecting the body or mind, so are an illness not a injury.
        Injury: False

        1. An injury is defined as damage inflicted on the body.
        In one sentence, explain whether "{self.input}" contains a phrase which aligns with the definition of an injury.
        2. If the input contains an example of an injury, give the injury, else write Injury: False.

        Use the following output format:
        Explanation: <your explanation>
        Injury: <your injury>'''
    
class Illness(PromptInput):
    def __init__(self, input):
        super().__init__()
        self.input = input
        self.pattern_matching_method = 'extract_illness'

    def get_field_checked(self):
        return 'How it harms'
    
    def generate_prompt(self):
        return f'''

        Follow these instructions:
        1. An illness is defined as a disease or sickness affecting the body or mind.
        In one sentence, explain whether the input contains a phrase which aligns with the definition of an illness.
        2. If the input contains an example of an illness, give the illness, else write Illness: False.

        Input: A worker falling from a height
        Explanation: Explanation: A worker falling from a height is a description of the event which leads to harm, not the harm itself.
        Illness: False

        Input: It results in a burn.
        Explanation: A burn is not a disease or sickness affecting the body or mind.
        Illness: False

        Input: Prolonged exposure to high temperatures without proper precautions
        Explanation: Prolonged exposure to high temperatures without proper precautions is a description of the event which leads to an harm, not the harm itself.
        Illness: False

        Input: Foodborne illnesses due to improper handling of food items
        Explanation: Foodborne illnesses are a type of disease or sickness affecting the body or mind.
        Illness: Foodborne illnesses

        1. An illness is defined as a disease or sickness affecting the body or mind.
        In one sentence, explain whether "{self.input}" contains a phrase which aligns with the definition of an illness.
        2. If the input contains an example of an illness, give the illness, else write Illness: False.

        Use the following output format:
        Explanation: <your explanation>
        Illness: <your harm>'''

# class EventCausedByHazard(PromptInput):
#     def __init__(self, hazard, hazard_event):
#         super().__init__()
#         self.hazard = hazard
#         self.hazard_event = hazard_event

#         self.pattern_matching_method = 'check_string_for_true_or_false_with_no_overall_answer'
    
#     def get_field_checked(self):
#         return 'Event that leads to harm'
    
#     def generate_prompt(self):
#         return f'''
#         Follow these instructions:
#         Is "{self.hazard_event}" an event caused by "{self.hazard}"?
#         If it is, answer True, else answer False.

#         Input:
#         Is "A worker falling from a height" an event caused by "Unsecured scaffolding"?
#         Answer: True

#         Input:
#         Is "Person slipping on wet floor" an event caused by "Unsecured scaffolding"?
#         Answer: False

#         Input:
#         Is "Chef burning hand on stove" an event caused by "Hot surfaces"?
#         Answer: True

#         Input:
#         Is "Swimmer drowning" an event caused by "Reckless drivers"?
#         Answer: False
        
#         Input:
#         Is "{self.hazard_event}" an event caused by "{self.hazard}"?
        
#         Use the following output format:
#         Answer: <your answer>'''

# class HarmCausedByEvent(PromptInput):
#     def __init__(self, hazard_event, how_it_harms):
#         super().__init__()
#         self.hazard_event = hazard_event
#         self.how_it_harms = how_it_harms

#         self.pattern_matching_method = 'check_string_for_true_or_false_with_no_overall_answer'
    
#     def get_field_checked(self):
#         return 'How it harms'
    
#     def generate_prompt(self):
#         return f'''
#         Follow these instructions:
#         Does "{self.hazard_event}" cause "{self.how_it_harms}"?
#         If it does, answer True, else answer False.

#         Input:
#         Does "A worker falling from a height" cause harm by "Impact injury"?
#         Answer: True

#         Input:
#         Does "Person slipping on wet floor" cause harm by "Impact injury"?
#         Answer: False

#         Input:
#         Does "Chef burning hand on stove" cause harm by "Burns"?
#         Answer: True

#         Input:
#         Does "Swimmer drowning" cause harm by "Impact injury"?
#         Answer: False
        
#         Input:
#         Does "{self.hazard_event}" cause harm by "{self.how_it_harms}"?
        
#         Use the following output format:
#         Answer: <your answer>'''
    
class HazardEvent(PromptInput):
    def __init__(self, activity, hazard, who_it_harms, how_it_harms):
        super().__init__()
        self.activity = activity
        self.hazard = hazard
        self.who_it_harms = who_it_harms
        self.how_it_harms = how_it_harms
        self.pattern_matching_method = "extract_hazard_event"

    def generate_prompt(self, harm_caused):
        return f'''
        Follow these instructions:
        Given the activity: "{self.activity}", the hazard: "{self.hazard}", who it harms: {self.who_it_harms}, and how it harms: "{self.how_it_harms}":
        1. The harm caused is: "{harm_caused}".
        2. Describe the events which lead to the harm caused? Don't mention the harm caused: "{harm_caused}".

        Example Output:
        Activity: Building a house
        Hazard: Unsecured scaffolding
        Who it harms: Construction workers
        How it harms: Impact injuries caused by falling from a height
        Harm caused: Impact injury
        Event that leads to harm: A worker falling from a height

        Example Output:
        Activity: Fluids Laboratory
        Hazard: Syringes with sharp needles
        Who it harms: Students
        How it harms: Sharp needles can pierce the skin and cause bleeding
        Harm caused: Puncture wound
        Event that leads to harm: A student being pricked by the syringe needle

        Use the following output format:
        Activity: {self.activity}
        Hazard: {self.hazard}
        Who it harms: {self.who_it_harms}
        How it harms: {self.how_it_harms}
        Harm caused: <your harm>
        Event that leads to harm: <your event>'''

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
    
        #     1. Description: In one sentence, describe the hazard: "{self.hazard}" during the
        # activity: "{self.activity}" given how the hazard harms: "{self.how_it_harms}"
        # and who the hazard harms: "{self.who_it_harms}".
    
    def generate_prompt_without_few_shot_examples(self):
        return f'''Follow these instructions:
        2. Reduces Harm Explanation: Explain whether "{self.control_measure}" has the potential to reduce the harm caused by the hazard: "{self.hazard}" given how it harms: "{self.how_it_harms}".
        3. Reduces Harm Answer: If "{self.control_measure}" reduces the harm, answer True, else answer False.
        1. Protective Clothing Explanation: Protective clothing provides a physical barrier between the person and potential hazards. In one sentence, explain whether "{self.control_measure}" is an example of protective clothing
        2. Protective Clothing Answer: If "{self.control_measure}" is an example of providing protective clothing, answer True, else answer False.
        3. Part of Body: Which parts of the body of the "{self.who_it_harms}" are most likely damaged by the hazard: "{self.hazard}" given how it harms: "{self.how_it_harms}"? 
        4. Protects Part of Body Explanation: Explain whether "{self.control_measure}" protects some of these body parts from harm.
        5. Protects Part of Body Answer: If "{self.control_measure}" protects some of these body parts, answer True, else answer False.
        6. If both "Reduces Harm", "Protective Clothing Answer" and "Protects Part of Body Answer" are True, answer True. Else answer False.'''
    
    def generate_prompt(self):
        example_of_correct_protective_clothing = '''
        Example Input:
        1. Reduces Harm Explanation: Explain whether "Wearing lab coat and PPE" has the potential to reduce the harm caused by the hazard: "Syringes with sharp needles" given how it harms: "Sharp needles can pierce the skin and cause bleeding"?
        2. Reduces Harm Answer: If "Wearing lab coat and PPE" reduces the harm, answer True, else answer False.
        1. Protective Clothing Explanation: Protective clothing provides a physical barrier between the person and potential hazards. In one sentence, explain whether ""Wearing lab coat and PPE"" is an example of protective clothing
        2. Protective Clothing Answer: If "Wearing lab coat and PPE" is an example of providing protective clothing, answer True, else answer False.
        3. Part of Body: Which parts of the body of the "Students" are most likely damaged by the hazard: "Syringes with sharp needles" given how it harms: "Sharp needles can pierce the skin and cause bleeding"? 
        4. Protects Part of Body Explanation: Explain whether "Wearing lab coat and PPE" protects some of these body parts from harm.
        5. Protects Part of Body Answer: If "Wearing lab coat and PPE" protects some of these body parts, answer True, else answer False.
        6. If both "Protective Clothing Answer" and "Protects Part of Body Answer" are True, answer True. Else answer False.

        Output:
        Reduces Harm Explanation: "Wearing lab coat and PPE" reduces the harm caused by the hazard as it provides a physical barrier between the person and the sharp syringe needle.
        Reduces Harm Answer: True.
        Protective Clothing Explanation: Yes, wearing a lab coat is a prime example of wearing protective clothing, as it shields individuals from potential hazards in the laboratory.
        Protective Clothing Answer: True.
        Part of Body: "Syringes with sharp needles" can pierce the skin and cause bleeding on any body part.
        Protects Part of Body Explanation: "Wearing lab coat and PPE" provides protection for the arms and body so provides protection for some of these body parts.
        Protects Part of Body Answer: True. 
        Overall Answer: True.

        Example Input:
        1. Reduces Harm Explanation: Explain whether "Metal gloves" has the potential to reduce the harm caused by the hazard: "Sharp blade" given how it harms: "Cutting injury"?
        2. Reduces Harm Answer: If "Metal gloves" reduces the harm, answer True, else answer False.
        1. Protective Clothing Explanation: Protective clothing provides a physical barrier between the person and potential hazards. In one sentence, explain whether "Metal gloves" is an example of protective clothing
        2. Protective Clothing Answer: If "Metal gloves" is an example of providing protective clothing, answer True, else answer False.
        3. Part of Body: Which parts of the body of the "Operator" are most likely damaged by the hazard: "Sharp blade" given how it harms: "Cutting injury"?
        4. Protects Part of Body Explanation: Explain whether "Metal gloves" protects some of these body parts from harm.
        5. Protects Part of Body Answer: If "Metal gloves" protects some of these body parts, answer True, else answer False.
        6. If both "Protective Clothing Answer" and "Protects Part of Body Answer" are True, answer True. Else answer False.

        Output:
        Reduces Harm Explanation: "Metal gloves" reduces the harm caused by the hazard as they provide a physical barrier between the operator's hand and the sharp blade.
        Reduces Harm Answer: True.
        Protective Clothing Explanation: "Metal gloves" are an example of protective clothing as they provide a physical barrier between the person and potential hazards.
        Protective Clothing Answer: True.
        Part of Body: "Sharp blade" can cause cutting injuries to the hands and fingers.
        Protects Part of Body Explanation: "Metal gloves" protect the hands and fingers from harm.
        Protects Part of Body Answer: True.
        Overall Answer: True.

        Example Input:
        1. Reduces Harm Explanation: Explain whether "Wearing gloves" has the potential to reduce the harm caused by the hazard: "Horse Kicks" given how it harms: "Impact injury"?
        2. Reduces Harm Answer: If "Wearing gloves" reduces the harm, answer True, else answer False.
        1. Protective Clothing Explanation: Protective clothing provides a physical barrier between the person and potential hazards. In one sentence, explain whether "Wearing gloves" is an example of protective clothing
        2. Protective Clothing Answer: If "Wearing gloves" is an example of providing protective clothing, answer True, else answer False.
        3. Part of Body: Which parts of the body of the "Stable hand" are most likely damaged by the hazard: "Horse Kicks" given how it harms: "Impact injury"?
        4. Protects Part of Body Explanation: Explain whether "Wearing gloves" protects some of these body parts from harm.
        5. Protects Part of Body Answer: If "Wearing gloves" protects some of these body parts, answer True, else answer False.
        6. If both "Protective Clothing Answer" and "Protects Part of Body Answer" are True, answer True. Else answer False.

        Output:
        Reduces Harm Explanation: "Wearing gloves" does not reduce the harm caused by the hazard as it will not protect the stable hand from a horse's kick.
        Reduces Harm Answer: False.
        Protective Clothing Explanation: "Wearing gloves" is an example of protective clothing as it provides a physical barrier between the person and potential hazards.
        Protective Clothing Answer: True.
        Part of Body: "Horse Kicks" can cause significant impact injuries to the torso, back and head.
        Protects Part of Body Explanation: "Wearing gloves" does not protect the torso, back and head from harm.
        Protects Part of Body Answer: False.
        Overall Answer: False.
        '''

        return f'''

        {example_of_correct_protective_clothing}

        {self.generate_prompt_without_few_shot_examples()}

        Use the following output format:
        Reduces Harm Explanation: <your explanation>
        Reduces Harm Answer: <your answer>
        Protective Clothing Explanation: <your description>
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
    
    def get_longform_feedback(self, prompt_output=''):
        return f"{self.control_measure} is an example of wearing protective clothing, which reduces the harm caused by the hazard event so is therefore a mitigation measure."
    
    # TODO: When you have hazard event input, can include below
    def get_recommendation(self):
        return f"""For the prevention field, enter a control measure which reduces the likelihood of the hazard event. Wearing protective clothing does not reduce the likelihood of the hazard event so it is not a prevention measure."""
    
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
        2. Reduces Harm Explanation: Explain whether "{self.control_measure}" has the potential to reduce the harm caused by the hazard: "{self.hazard}" given how it harms: "{self.how_it_harms}".
        3. Reduces Harm Answer: If "{self.control_measure}" reduces the harm, answer True, else answer False.
        4. After Medical Emergency Explanation: Explain whether "{self.control_measure}" is likely to be done after a medical emergency.
        5. After Medical Emergency Answer: If "{self.control_measure}" is likely to be done after a medical emergency, answer True, else answer False.
        6. Initial Medical Response Explanation: "Definition of initial medical response": "Immediate actions taken by individuals in the presence of a medical emergency or injury to provide basic care or initiate further assistance as needed.
        Given this "Definition of initial medical response", explain whether "{self.control_measure}" is an example of initial medical response.
        7. Initial Medical Response Answer: If "{self.control_measure}" is an example of an example of initial medical response, answer True, else answer False.
        8. If all of "After Medical Emergency Answer", "First Aid Answer" and "Reduces Harm Answer" are True, answer True, else answer False.'''
    
    def generate_prompt(self):
        example_of_correct_protective_clothing = '''
        Example Input:
        1. Description: Describe "Washing the eyes out with clean water" in one sentence.
        4. Reduces Harm Explanation: Explain whether "Washing the eyes out with clean water" has the potential to reduce the harm caused by the hazard: "Ink spillage" given how it harms: "Serious eye damage".
        5. Reduces Harm Answer: If "Washing the eyes out with clean water" reduces the harm, answer True, else answer False.
        2. After Medical Emergency Explanation: Explain whether "Washing the eyes out with clean water" is likely to be done after a medical emergency.
        3. After Medical Emergency Answer: If "Washing the eyes out with clean water" is likely to be done after a medical emergency, answer True, else answer False.
        2. Initial Medical Response Explanation: "Definition of initial medical response": "Immediate actions taken by individuals in the presence of a medical emergency or injury to provide basic care or initiate further assistance as needed.
        Given this "Definition of initial medical response", explain whether ""Washing the eyes out with clean water"" is an example of initial medical response.
        3. Initial Medical Response Answer:  If "Washing the eyes out with clean water" is an example of initial medical response, answer True, else answer False.
        6. If all of "After Medical Emergency Answer", "First Aid Answer" and "Reduces Harm Answer" are True, answer True, else answer False.

        Output:
        Description: "Washing the eyes out with clean water" is the process of rinsing the eyes with clean water to remove any foreign bodies or chemicals that may have entered the eye.
        Reduces Harm Explanation: "Washing the eyes out with clean water" reduces the harm caused by the hazard as it helps to wash the ink out of the eyes and reduce eye damage.
        Reduces Harm Answer: True.
        After Medical Emergency Explanation: "Washing the eyes out with clean water" is likely to be done after a medical emergency as it is a first aid measure.
        After Medical Emergency Answer: True.
        Intial Medical Response Explanation: "Washing the eyes out with clean water" is an example of basic care provided to an individual so is an example of "initial medical response".
        Initial Medical Response Answer: True.
        Overall Answer: True.

        Example Input:
        1. Description: Describe "Call for urgent medical assistance" in one sentence.
        4. Reduces Harm Explanation: Explain whether "Call for urgent medical assistance" has the potential to reduce the harm caused by the hazard: "Ink spillage" given how it harms: "Serious eye damage".
        5. Reduces Harm Answer: If "Call for urgent medical assistance" reduces the harm, answer True, else answer False.
        2. After Medical Emergency Explanation: Explain whether "Call for urgent medical assistance" is likely to be done after a medical emergency.
        3. After Medical Emergency Answer: If "Call for urgent medical assistance" is likely to be done after a medical emergency, answer True, else answer False.
        2. Initial Medical Response Explanation: "Definition of initial medical response": "Immediate actions taken by individuals in the presence of a medical emergency or injury to provide basic care or initiate further assistance as needed.
        Given this "Definition of initial medical response", explain whether "Call for urgent medical assistance" is an example of initial medical response.
        3. Initial Medical Response Answer:  If "Call for urgent medical assistance" is an example of initial medical response, answer True, else answer False.
        6. If all of "After Medical Emergency Answer", "First Aid Answer" and "Reduces Harm Answer" are True, answer True, else answer False.

        Output:
        Description: "Call for urgent medical assistance" is the process of contacting emergency services to provide urgent medical care to an individual.
        Reduces Harm Explanation: "Call for urgent medical assistance" has the potential to reduce the harm of "Serious eye damage" as the medical assistance will provide the necessary care to reduce the harm caused by the hazard.
        Reduces Harm Answer: True.
        After Medical Emergency Explanation: "Call for urgent medical assistance" is likely to be done after a medical emergency.
        After Medical Emergency Answer: True.
        Intial Medical Response Explanation: "Call for urgent medical assistance" is an example of initiating further assistance as needed so is an example of "initial medical response".
        Initial Medical Response Answer: True.
        Overall Answer: True.

        Example Input:
        Describe "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." in one sentence.
        After Medical Emergency Explanation: Explain whether "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." is likely to be done after a medical emergency.
        Reduces Harm Explanation: Explain whether "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." has the potential to reduce the harm caused by the hazard: "Climbing Protection Gear (Cams and Hexs)" given how it harms: "Some equipment is heavy so could hurt if dropped on feet.".
        Reduces Harm Answer: If "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." reduces the harm, answer True, else answer False.
        After Medical Emergency Answer: If "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." is likely to be done after a medical emergency, answer True, else answer False.
        Initial Medical Response Explanation: "Definition of initial medical response": "Immediate actions taken by individuals in the presence of a medical emergency or injury to provide basic care or initiate further assistance as needed.
        Given this "Definition of initial medical response", explain whether "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." is an example of initial medical response.
        Initial Medical Response Answer:  If "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." is an example of initial medical response, answer True, else answer False.
        If all of "After Medical Emergency Answer", "First Aid Answer" and "Reduces Harm Answer" are True, answer True, else answer False.

        Output:
        Description: "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." is the process of providing information to individuals who wish to hold the equipment of the risk and demonstrate how they are used correctly.
        Reduces Harm Explanation: "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." has the potential to reduce the harm caused by the hazard of "Climbing Protection Gear (Cams and Hexs)" as it ensures that individuals are aware of the risks and know how to use the equipment properly, reducing the likelihood of accidents or injuries.
        Reduces Harm Answer: True.
        After Medical Emergency Explanation: "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." will not be done after a medical emergency as it will be done before the equipment is used.
        After Medical Emergency Answer: False.
        Intial Medical Response Explanation: "Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly." is not an example of initial medical response as it does not involve providing immediate care or assistance in a medical emergency.
        Initial Medical Response Answer: False.
        Overall Answer: False.

        '''

        return f'''

        {example_of_correct_protective_clothing}

        {self.generate_prompt_without_few_shot_examples()}

        Use the following output format:
        Description: <your description>
        Reduces Harm Explanation: <your explanation>
        Reduces Harm Answer: <your answer>
        After Medical Emergency Explanation: <your explanation>
        After Medical Emergency Answer: <your answer>
        Initial Medical Response Explanation: <your explanation>
        Initial Medical Response Answer: <your answer>
        Overall Answer: <your answer>'''
    
    def get_shortform_feedback(self, feedback_type):
        # Feedback is only given when a mitigation is written in prevention input.

        if feedback_type == 'positive':
            return f"Correct! '{self.control_measure}' is an example of a mitigation measure"
        if feedback_type == 'negative':
            return f"Incorrect. '{self.control_measure}' is not a prevention measure, but is actually a mitigation measure."
    
    def get_longform_feedback(self, prompt_output=''):
        return f"""{self.control_measure} is an example of a first aid measure, which reduces the harm caused by the hazard event after it has occurred; it is therefore a mitigation measure."""

    # TODO: When you have hazard event input, can include in feedback.
    def get_recommendation(self):
        return f"""For the prevention field, enter a control measure which reduces the likelihood of the hazard event. First aid is applied after the hazard event so does not reduce the likelihood of the hazard event occurring and is therefore not a prevention measure."""

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
        1. In one sentence, describe the hazard: '{self.hazard}' during the
        activity: '{self.activity}' given how the hazard causes harm: '{self.how_it_harms}'.
        2. Explain whether or not '{self.prevention}' reduces the likelihood that the hazard occurs.
        If so, it is a prevention measure.
        3. If the hazard occurs, explain whether or not '{self.prevention}' removes or reduces the chance of {self.how_it_harms}.
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
        1. In one sentence, describe the hazard: 'Ink spillage on students face' during the
        activity: 'Fluids laboratory' given how the hazard causes harm: 'Serious eye damage'.
        3. Explain whether or not 'First aid' reduces the likelihood that the hazard occurs.
        If so, it is a prevention measure.
        4. Assuming the hazard occurs, explain whether or not 'First aid' removes or reduces the chance of 'Serious eye damage'.
        If so, it is a mitigation measure.
        5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        prevention measure and a mitigation measure, answer 'Both'.

        Output: 
        Hazard Description: The hazard of 'Ink spillage on student's face' during the activity 'Fluids laboratory' can lead to serious eye damage to students.
        Prevention Explanation: 'First aid' is a reactive measure applied after the hazard of 'Ink spillage on student's face'; it therefore does not reduce the likelihood of the hazard and is not a prevention measure.
        Mitigation Explanation: If ink has been spilled onto a student's face, 'first aid' will help to wash the ink out of the eyes and reduce eye damage after the hazard has occurred; as it reduces the harm caused by the hazard, it is therefore a mitigation measure.
        Answer: Mitigation.

        Follow these
        Input: instructions:
        1. In one sentence, describe the hazard: 'Water being spilt on the floor causing students to slip' during the
        activity: 'Fluids laboratory' given how the hazard harms: 'Impact injury'.
        3. Explain whether or not 'Do not move the water tank when it is full' reduces the likelihood that the hazard occurs.
        If so, it is a prevention measure.
        4. Assuming the hazard occurs, explain whether or not 'Do not move the water tank when it is full' removes or reduces the chance of 'Impact injury'.
        If so, it is a mitigation measure.
        5. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a        
        prevention measure and a mitigation measure, answer 'Both'.

        Hazard Description: The hazard of 'Water being spilt on the floor causing students to slip' during the activity 'Fluids laboratory' can lead to impact injuries.
        Prevention Explanation: 'Keeping the water tank stationary when it's full' means water cannot be spilled on to the floor by moving the water tank; no water on the floor reduces the likelihood of the student slipping; since it reduces the likelihood of the hazard, it is a prevention measure.
        Mitigation Explanation: If water has been spilled on the floor, 'not moving the water tank when it is full' does not remove or reduce the harm caused by the hazard, as the water is already spilled to pose a slipping hazard; as it does not reduce the harm caused by the hazard, it is not a mitigation measure.
        Answer: Prevention.

        Follow these instructions:
        1. In one sentence, describe the hazard: 'Cut Zip tie flies and hits audience member' during the
        activity: 'Using a spring contraption as a demonstration for a TPS presentation' given how the hazard harms: 'Impact injury.'.
        2. Explain whether or not 'Keep hand around zip tie when cutting to stop it from flying' reduces the likelihood that the hazard occurs.
        If so, it is a prevention measure.
        3. If the hazard occurs, explain whether or not 'Keep hand around zip tie when cutting to stop it from flying' removes or reduces the chance of Impact injury..
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a migitation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a
        prevention measure and a mitigation measure, answer 'Both'.

        Hazard Description: The hazard of 'Cut Zip tie flies and hits audience member' during the activity 'Using a spring contraption as a demonstration for a TPS presentation' can lead to impact injuries.
        Prevention Explanation: 'Keeping hand around zip tie when cutting to stop it from flying' will stop the zip tie from flying and therefore stop the hazard from occurring. Therefore, the likelihood of the hazard occurring has been reduced to zero; since the likelihood has been reduced, it is therefore a prevention measure.
        Mitigation Explanation: If the hazard occurs and the zip tie flies and hits an audience member, 'keeping hand around zip tie when cutting to stop it from flying' does not remove or reduce the impact injury caused by the hazard, as the zip tie has already flown and caused harm; it is therefore not a mitigation measure.
        Answer: Prevention.
        """

        return f'''
        {all_few_shot_examples}

        {self.generate_prompt_without_few_shot_examples()}

        Use the following output format:
        Hazard Description: <your hazard description>
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
    
    def get_longform_feedback(self, prompt_output='', pattern_to_search_for='Prevention Explanation'):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, pattern_to_search_for)

    # TODO: When you have hazard event input, can include in feedback.
    def get_recommendation(self, recommendation_type):
        if recommendation_type == 'neither':
            return "For the prevention field, enter a control measure which reduces the likelihood of the hazard event."
        
        if recommendation_type == 'misclassification':
            return f"""A mitigation measure reduces the harm caused by the hazard event either while the hazard event is occurring or after it has occurred. On the other hand, a prevention measure reduces the likelihood of the hazard event occurring in the first place. Please use the above definitions to ammend your prevention input."""
    
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
    
    def get_longform_feedback(self, prompt_output='', pattern_to_search_for='Mitigation Explanation'):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.extract_section_of_prompt_until_new_line_or_end_of_string(prompt_output, pattern_to_search_for)
    
    # TODO: When you have hazard event input, can include in feedback.
    def get_recommendation(self, recommendation_type):
        if recommendation_type == 'neither':
            return "For the mitigation field, enter a control measure which reduces the harm caused by the hazard event either while the hazard event is occurring or after it has occurred."
        
        if recommendation_type == 'misclassification':
            return f"""A prevention measure reduces the likelihood of the hazard event occurring in the first place. On the other hand, a mitigation measure reduces the harm caused by the hazard event while it is happening or after it has occurred. Please use the above definitions to ammend your mitigation input."""