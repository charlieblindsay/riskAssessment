try:
    from RegexPatternMatcher import RegexPatternMatcher
except:
    from .RegexPatternMatcher import RegexPatternMatcher

class PromptInput:
    def __init__(self):
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
        
class NoInformationProvided(PromptInput):
    def __init__(self, input: str):
        super().__init__()
        self.pattern_matching_method = 'check_string_for_no_information_provided'
        self.candidate_labels = ['control measure', 'no information provided']
        self.input = input
    
    def generate_prompt(self):
        return f'''
        Follow these instructions:
        1. Classify the following input as either "No information provided" or "Control measure"

        <EXAMPLES>
        Input: "N/A" 
        Answer: No information provided

        Input: "Not Applicable" 
        Answer: No information provided

        Input: "Unknown" 
        Answer: No information provided

        Input: "None" 
        Answer: No information provided

        Input: "TBD" 
        Answer: No information provided

        Input: "To Be Determined" 
        Answer: No information provided

        Input: "Unspecified" 
        Answer: No information provided

        Input: "No Information Available" 
        Answer: No information provided

        Input: "Do Not Know" 
        Answer: No information provided

        Input: "Not specified" 
        Answer: No information provided

        Input: "Unavailable" 
        Answer: No information provided

        Input: "Not applicable" 
        Answer: No information provided

        Input: "Not known" 
        Answer: No information provided

        Input: "Wear helmet"
        Answer: Control measure

        Input: "Take care"
        Answer: Control measure

        Input: "Apply ice"
        Answer: Control measure
        </EXAMPLES>

        <OUTPUT FORMAT>
        Use the following output format:
        Answer: <your answer>
        </OUTPUT FORMAT>

        Input: "{self.input}"'''

class Activity(PromptInput):
    def __init__(self, activity: str):
        super().__init__()
        self.activity = activity

    def get_field_checked(self):
        return 'Activity'
    
        # If "Golf" is an example of an activity, answer True, else answer False.

        # Use the following output format:
        # Overall Answer: <your answer>
    
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
        1. In one sentence, describe the hazard: '{self.hazard}' during the activity: '{self.activity}'.
        2. In one sentence, explain whether or not '{self.how_it_harms}' is a way that this hazard directly causes harm. 
        3. If it is, answer True, else answer False.
        '''
    
    def generate_prompt(self):
        example_of_correct_how_it_harms = f'''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Electrocution' during the activity: 'Fluids laboratory'.
        2. In one sentence, explain whether or not 'Electrocuted by mains voltage' is a way that this hazard DIRECTLY causes harm. 
        3. If it is, answer True, else answer False.

        Output:
        Description: 'Electrocution' during a fluids laboratory can occur when an individual comes into contact with mains voltage.
        Explanation: As water is a conductor of electricity, touching electronics with wet hands can DIRECTLY cause harm
        through electrocution as the water provides a path for electrical current to flow through the body.
        Overall Answer: True
        '''

        example_of_incorrect_how_it_harms = f'''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Volcanic eruption' during the activity: 'Volcano visit'.
        2. In one sentence, explain whether or not "Heavy impact when falling onto a demonstrator and causing injury" is a way that this hazard DIRECTLY causes harm.
        3. If it is, answer True, else answer False.

        Output:
        Description: A volcanic eruption during a volcano activity can lead to various hazards and risks.
        Explanation: "Heavy impact when falling onto a demonstrator and causing injury" is not a DIRECT way that a volcanic eruption causes harm. The primary dangers of a volcanic eruption include lava flows, ash clouds, and pyroclastic flows.
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
    def __init__(self, who_it_harms, activity, hazard, how_it_harms):
        super().__init__()
        self.who_it_harms = who_it_harms
        self.activity = activity
        self.hazard = hazard
        self.how_it_harms = how_it_harms

    def get_field_checked(self):
        return 'Who It Harms'
    
    def generate_prompt(self):
        
        example_of_correct_who_it_harms = f'''
        Example Input:
        Follow these instructions:
        1. In one sentence, describe the hazard: 'Getting kicked by a horse' during the activity: 'Mucking out stable', given how it harms: 'Impact injury'.
        2. In one sentence, explain whether it is possible that a 'Stable hand' is harmed by this hazard.
        3. If it is possible, answer True, else answer False.

        Output:
        Description: When mucking out a stable, it is possible that the person mucking out is kicked by a horse, resulting in impact injuries.
        Explanation: It is likely that a "Stable hand" would muck out a stable and therefore possible for the horse to kick the "Stable hand", causing an impact injury.
        Overall Answer: True
        '''

        example_of_incorrect_who_it_harms = f'''
        Example Input:
        1. In one sentence, describe the hazard: 'Ink spillage' during the activity 'Fluids laboratory', given how it harms: 'Serious eye damage.
        2. In one sentence, explain whether it is possible that a 'Stable hand' is harmed by this hazard.
        3. If it is possible, answer True, else answer False.

        Output:
        Description: Ink spillage during the fluids laboratory activity can cause serious eye damage if the ink comes into contact with the eyes.
        Explanation: It is unlikely that a stable hand would be harmed by this hazard, as they typically work with horses in a stable environment and are not involved in laboratory activities involving fluids like ink.
        Overall Answer: False
        '''
        return f'''
        {example_of_correct_who_it_harms}

        {example_of_incorrect_who_it_harms}

        Follow these instructions:
        1. In one sentence, describe the hazard: '{self.hazard}' during the activity: '{self.activity}', given how it harms: '{self.how_it_harms}'.
        2. In one sentence, explain whether it is possible that '{self.who_it_harms}' is harmed by this hazard.
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

class HarmCausedAndHazardEvent(PromptInput):
    def __init__(self, activity, hazard, how_it_harms, who_it_harms):
        super().__init__()
        self.activity = activity
        self.hazard = hazard
        self.how_it_harms = how_it_harms
        self.who_it_harms = who_it_harms
        self.pattern_matching_method = 'extract_harm_caused_and_hazard_event'

    def generate_prompt_without_few_shot_examples(self):
        return f'''
        1. Describe the harm caused: '{self.how_it_harms}' by the hazard: '{self.hazard}' during the activity: '{self.activity}'.
        The harm caused by a hazard refers to the negative consequences resulting from the hazard event.
        2. From the above description, infer the harm caused. NOTE: This is not an event, but a consequence of an event.
        3. Describe the events which lead to the harm caused. Don't refer to the harm caused.
        '''

    # def generate_prompt(self):

    #     return f'''

    #     <EXAMPLE INSTRUCTIONS>
    #     1. The harm caused by a hazard refers to the negative consequences resulting from the hazard event. 
    #     Describe the harm caused: 'Could injure' for who it harms: 'Cyclists' by the hazard: 'Getting hit' during the activity: 'Riding a Bike' .
    #     Write your answer in the format: <activity> can lead to <hazard event>, which could result in <harm caused> for <who it harms>.
    #     2. Thinking step by step, what are the series of events which lead to harm.
    #     3. The "harm caused" is the negative consequences from the hazard event. Infer the "harm caused" to 'Cyclists' by this series of events.
    #     4. Now write the series of events that lead to harm. NOTE: Do not mentioning the "harm caused".
    #     </EXAMPLE INSTRUCTIONS>

    #     <EXAMPLE OUTPUT>
    #     Description: Riding a bike can lead to getting hit, which could result in injuring cyclists.
    #     Series of events with harm caused: 
    #     - The cyclist is riding their bike
    #     - A vehicle approaches the cyclist
    #     - The vehicle collides with the cyclist
    #     - The cyclist is injured in the collision
    #     Harm caused: The cyclist is injured in the collision
    #     Event that leads to harm: A cyclist is riding their bike, a vehicle approaches the cyclist, the vehicle collides with the cyclist.

    #     </EXAMPLE OUTPUT>

    #     <EXAMPLE INSTRUCTIONS>
    #     1. The harm caused by a hazard refers to the negative consequences resulting from the hazard event. 
    #     Describe the harm caused: 'Mistakes by cyclists or motorists leading to crash' for who it harms: 'cyclists' by the hazard: 'Head injury' during the activity: 'Cycle commuting'.
    #     Write your answer in the format: <activity> can lead to <hazard event>, which could result in <harm caused> for <who it harms>.
    #     2. Thinking step by step, write the "series of events" which lead to this harm caused as a series of bullet points.
    #     3. The "harm caused" is the negative consequences from the hazard event. Infer the "harm caused" to 'cyclists' by this series of events.
    #     4. Now write the "series of events" but don't include the final bullet point.
    #     </EXAMPLE INSTRUCTIONS>

    #     <EXAMPLE OUTPUT>
    #     Description: Cycle commuting can lead to mistakes by cyclists or motorists leading to crash, which could result in head injury for cyclists.
    #     Series of events with harm caused: 
    #     - The cyclist is commuting by bike on the road
    #     - A motorist or the cyclist makes a mistake while driving/riding
    #     - This mistake leads to the cyclist and vehicle colliding
    #     - The collision causes the cyclist to fall off their bike
    #     - The cyclist's head impacts the ground or another hard surface
    #     - The cyclist sustains a head injury
    #     Harm caused: The cyclist sustains a head injury
    #     Event that leads to harm: The cyclist is commuting by bike on the road, a motorist or the cyclist makes a mistake while driving/riding, this mistake leads to the cyclist and vehicle colliding, the collision causes the cyclist to fall off their bike, the cyclist's head impacts the ground or another hard surface.
    #     </EXAMPLE OUTPUT>

    #     <EXAMPLE INSTRUCTIONS>
    #     1. The harm caused by a hazard refers to the negative consequences resulting from the hazard event. 
    #     Describe the harm caused: 'When cut the zip tie may hit an audience member' for who it harms: 'audience' by the hazard: 'Cut Zip tie may fly' during the activity: 'Using a spring contraption as a demonstration for a TPS presentation'.
    #     Write your answer in the format: <activity> can lead to <hazard event>, which could result in <harm caused> for <who it harms>.
    #     2. Thinking step by step, write the "series of events" which lead to this harm caused as a series of bullet points.
    #     3. The "harm caused" is the negative consequences from the hazard event. Infer the "harm caused" to 'audience member' by this series of events.
    #     4. Now write the "series of events" but don't include the final bullet point.
    #     </EXAMPLE INSTRUCTIONS>

    #     <EXAMPLE OUTPUT>
    #     Description: Using a spring contraption as a demonstration for a TPS presentation, cutting the zip tie might cause it to fly, which could result in the zip tie hitting an audience member.
    #     Series of events with harm caused: 
    #     - The presenter is using a spring contraption for a demonstration
    #     - The spring contraption is held together by a zip tie
    #     - The presenter cuts the zip tie to release the spring
    #     - The cut zip tie flies off the contraption at high speed
    #     - The zip tie projectile hits an audience member
    #     - The audience member is injured by the impact
    #     Harm caused: The audience member is injured by the impact
    #     Event that leads to harm: The presenter is using a spring contraption for a demonstration, the spring contraption is held together by a zip tie, the presenter cuts the zip tie to release the spring, the cut zip tie flies off the contraption at high speed, the zip tie projectile hits an audience member.
    #     </EXAMPLE OUTPUT>

    #     <INSTRUCTIONS>
    #     1. The harm caused by a hazard refers to the negative consequences resulting from the hazard event. 
    #     Describe the harm caused: '{self.how_it_harms}' for '{self.who_it_harms}' by the hazard: '{self.hazard}' during the activity: '{self.activity}'.
    #     Write your answer in the format: <activity> can lead to <hazard event>, which could result in <harm caused> for <who it harms>.
    #     2. Thinking step by step, write the "series of events" which lead to this harm caused as a series of bullet points.
    #     3. The "harm caused" is the negative consequences from the hazard event. Infer the "harm caused" for '{self.who_it_harms}' by this series of events.
    #     4. Now write the "series of events" but don't include the final bullet point.
    #     </INSTRUCTIONS>

    #     <OUTPUT FORMAT>
    #     Use the following output format:
    #     Description: <description>
    #     Series of events with harm caused: <event that leads to harm>
    #     Harm caused: <harm caused>
    #     Event that leads to harm: <event that leads to harm>
    #     </OUTPUT FORMAT>

    #     <OUTPUT>'''

    def generate_prompt(self):

        return f'''

        <EXAMPLE INSTRUCTIONS>
        1. The harm caused by a hazard refers to the negative consequences resulting from the hazard event. 
        Describe the harm caused: 'Could injure' for who it harms: 'Cyclists' by the hazard: 'Getting hit' during the activity: 'Riding a Bike' .
        Write your answer in the format: <activity> can lead to <hazard event>, which could result in <harm caused> for <who it harms>.
        2. Extract the harm caused.
        3. Describe the events which lead to this harm caused. Don't refer to the harm caused.
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Description: Riding a bike can lead to getting hit by a car, which could lead to an impact injury for cyclists.
        Harm caused: Impact injury
        Event that leads to harm: Getting hit by a car while riding a bike
        </EXAMPLE OUTPUT>

        <EXAMPLE INSTRUCTIONS>
        1. The harm caused by a hazard refers to the negative consequences resulting from the hazard event. 
        Describe the harm caused: 'Mistakes by cyclists or motorists leading to crash' for who it harms: 'cyclists' by the hazard: 'Head injury' during the activity: 'Cycle commuting'.
        Write your answer in the format: <activity> can lead to <hazard event>, which could result in <harm caused> for <who it harms>.
        2. Extract the harm caused.
        3. Describe the events which lead to this harm caused. Don't refer to the harm caused.
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Description: Cycle commuting can lead to mistakes by cyclists or motorists leading to a crash, which can cause a head injury.
        Harm caused: Head injury
        Event that leads to harm: Mistakes by cyclists or motorists leading to a crash
        </EXAMPLE OUTPUT>

        <EXAMPLE INSTRUCTIONS>
        1. The harm caused by a hazard refers to the negative consequences resulting from the hazard event. 
        Describe the harm caused: 'When cut the zip tie may hit an audience member' for who it harms: 'audience' by the hazard: 'Cut Zip tie may fly' during the activity: 'Using a spring contraption as a demonstration for a TPS presentation'.
        Write your answer in the format: <activity> can lead to <hazard event>, which could result in <harm caused> for <who it harms>.
        2. Extract the harm caused.
        3. Describe the events which lead to this harm caused. Don't refer to the harm caused.
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Description: Using a spring contraption as a demonstration for a TPS presentation can lead to a cut zip tie flying and hitting an audience member.
        Harm caused: Impact injury to audience member
        Event that leads to harm: Cut zip tie may fly and hit an audience member

        <INSTRUCTIONS>
        1. The harm caused by a hazard refers to the negative consequences resulting from the hazard event. 
        Describe the harm caused: '{self.how_it_harms}' for '{self.who_it_harms}' by the hazard: '{self.hazard}' during the activity: '{self.activity}'.
        Write your answer in the format: <activity> can lead to <hazard event>, which could result in <harm caused> for <who it harms>.
        2. Extract the harm caused.
        3. Describe the events which lead to this harm caused. Don't refer to the harm caused.
        </INSTRUCTIONS>

        <OUTPUT>'''
    
class RiskDomainClassification(PromptInput):
    def __init__(self, hazard, how_it_harms, who_it_harms):
        super().__init__()
        self.hazard = hazard
        self.how_it_harms = how_it_harms
        self.who_it_harms = who_it_harms

        self.pattern_matching_method = 'check_string_for_classification'
        self.candidate_labels = ['physical risk to individuals', 'environmental risk']

    # Other classes
        # - Terrist risk
        # - Accidental risk
        # - Biohazard risk
    
    def generate_prompt(self):
        return f'''
        Follow these instructions:
        1. Classify the following hazard: '{self.hazard}' given how it harms: '{self.how_it_harms}' and who it harms: '{self.who_it_harms}' into one of the following categories:
        - Physical risk to individuals
        - Environmental risk

        <EXAMPLE INSTRUCTIONS>
        Classify the following hazard: "Fire in a building" given how it harms: "Clothes catch on fire and cause a burn" and who it harms: "Building occupants" into one of the following categories:
        - Physical risk to individuals
        - Environmental risk
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Classification: Physical risk to individuals
        </EXAMPLE OUTPUT>

        <EXAMPLE INSTRUCTIONS>
        Classify the following hazard: "Volcanic eruption" given how it harms: "Volcanic eruptions can cause destruction of property, infrastructure, loss of life, and disruption of ecosystems" and who it harms: "Residents, infrastructure, and the environment" into one of the following categories:
        - Physical risk to individuals
        - Environmental risk
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Classification: Environmental risk
        </EXAMPLE OUTPUT>
        '''

class ControlMeasureClassification(PromptInput):
    def __init__(self, control_measure, activity, hazard, how_it_harms, who_it_harms):
        super().__init__()
        self.control_measure = control_measure
        self.activity = activity
        self.hazard = hazard
        self.how_it_harms = how_it_harms
        self.who_it_harms = who_it_harms

        self.pattern_matching_method = 'check_string_for_prevention_mitigation_or_neither'
        self.candidate_labels = ['prevention', 'mitigation', 'neither', 'both']
        self.labels_indicating_correct_input = ['prevention', 'both']
    
    def generate_prompt_without_few_shot_examples(self):
        # TODO: Decide whether it should be: 2. Thinking step by step and thinking through all possible causes of the hazard event: "<hazard_event>"
        return f'''Follow these instructions:
        1. In one sentence, describe the hazard given the hazard event: "<hazard event>" during the
        activity: '{self.activity}' given the harm caused: "<harm caused>" for {self.who_it_harms}.
        2. Thinking step by step, explain whether or not '{self.control_measure}' reduces the likelihood that hazard event: "<hazard event>" occurs.
        If so, it is a prevention measure.
        3. Thinking step by step, explain whether or not '{self.control_measure}' removes or reduces the harm caused: "<harm caused>" for the '{self.who_it_harms}'.
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a mitigation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a
        prevention measure and a mitigation measure, answer 'Both'.'''
    
    def generate_prompt(self, hazard_event, harm_caused):

        # TODO: Should alter the mitigation explanations - sometimes a mitigation measure is done to prepare for 
        # the hazard event, not just to reduce the harm caused by the hazard event.

        all_few_shot_examples = """
        <EXAMPLE INSTRUCTIONS>
        1. In one sentence, describe the hazard given the hazard event: "Ink spillage" during the
        activity: 'Fluids laboratory' given the harm caused: "Serious eye damage" for Students.
        2. Write the Hazard Event: "Ink spillage"
        3. Write the Harm Caused: "Serious eye damage".
        4. Thinking step by step, explain whether or not 'Cleaning the eyes out with water' reduces the likelihood that the hazard event: "Ink spillage" occurs.
        If so, it is a prevention measure.
        5. Thinking step by step, explain whether or not 'Cleaning the eyes out with water' removes or reduces the harm caused: "Serious eye damage" for the 'Students'.
        If so, it is a mitigation measure.
        6. If it is a prevention measure, answer 'Prevention'. If it is a mitigation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a
        prevention measure and a mitigation measure, answer 'Both'.
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Hazard Description: During a fluids laboratory, ink spillage on a student's face causes serious eye damage.

        Prevention Explanation: Thinking step by step:
        - Cleaning the eyes out with water is a response measure taken after the ink spillage has already occurred.
        - The act of cleaning the eyes with water does therefore not reduce the likelihood of the hazard event.
        - It is therefore not a prevention measure.

        Mitigation Explanation: Thinking step by step:
        - If ink spills on a student's face and gets into their eyes, immediately cleaning the eyes out with water can help flush out the ink.
        - Removing the ink from the eyes quickly can reduce the exposure time and minimize the potential for serious eye damage.
        - While cleaning the eyes with water may not completely eliminate the risk of eye damage, it can significantly reduce the severity of the harm caused.
        - It is therefore a mitigation measure.

        Answer: Mitigation
        </EXAMPLE OUTPUT>

        <EXAMPLE INSTRUCTIONS>
        1. In one sentence, describe the hazard given the hazard event: "Student slips on water" during the
        activity: 'Fluids laboratory' given the harm caused: "Injury caused by students slipping" for Students.
        2. Write the Hazard Event: "Student slips on water"
        3. Write the Harm Caused: "Injury caused by students slipping".
        4. Thinking step by step, explain whether or not 'Keeping the water tank stationary when it's full' reduces the likelihood that hazard event: "Student slips on water" occurs.
        If so, it is a prevention measure.
        5. Thinking step by step, explain whether or not 'Keeping the water tank stationary when it's full' removes or reduces the harm caused: "Injury caused by students slipping" for the 'Students'.
        If so, it is a mitigation measure.
        6. If it is a prevention measure, answer 'Prevention'. If it is a mitigation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a
        prevention measure and a mitigation measure, answer 'Both'.
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Hazard Description: During a fluids laboratory, water spills on the floor, causing students to slip and suffer injuries
        Hazard Event: Student slips on water
        Harm Caused: Injury caused by students slipping

        Prevention Explanation: Thinking step by step:
        - Keeping the water tank stationary when it's full prevents it from being accidentally knocked over or moved, which could cause water to spill.
        - Therefore, keeping the water tank stationary reduces the likelihood of water spilling onto the floor is significantly reduced.
        - Reducing the chances of water spillage reduces the likelihood of students slipping on the wet floor.
        - Therefore, keeping the water tank stationary when it's full reduces the likelihood of the hazard event from occurring. It is a prevention measure.

        Mitigation Explanation: Thinking step by step:
        - Keeping the water tank stationary when it's full does not directly reduce the harm caused to students if they do slip on spilled water.
        - The severity of the injuries sustained from slipping on a wet floor is not mitigated by the tank's stationary position.
        - Therefore, keeping the water tank stationary when it's full does not remove or reduce the harm caused by "Injury caused by students slipping" for the students. It is not a mitigation measure.

        Answer: Prevention
        </EXAMPLE OUTPUT>

        <EXAMPLE INSTRUCTIONS>
        1. In one sentence, describe the hazard given the hazard event: "Car crashes into the cyclist" during the
        activity: 'Cycle commuting' given the harm caused: "Head injury" for Cyclist.
        2. Write the Hazard Event: "Car crashes into the cyclist"
        3. Write the Harm Caused: "Head injury".
        4. Thinking step by step, explain whether or not 'Wear a helmet' reduces the likelihood that hazard event: "Car crashes into the cyclist" occurs.
        If so, it is a prevention measure.
        5. Thinking step by step, explain whether or not 'Wear a helmet' removes or reduces the harm caused: "Head injury" for the 'Cyclist'.
        If so, it is a mitigation measure.
        6. If it is a prevention measure, answer 'Prevention'. If it is a mitigation meausure, answer 'Mitigation'.   
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a
        prevention measure and a mitigation measure, answer 'Both'.
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Hazard Description: A cyclist commuting to work gets hit by a car, resulting in a head injury.
        Hazard Event: Car crashes into the cyclist
        Harm Caused: Head injury

        Prevention Explanation: Thinking step by step:
        - Wearing a helmet does not directly influence the likelihood of a cyclist getting hit by a car.
        - A helmet does not change the behavior of the cyclist or the driver of the car.
        - The presence or absence of a helmet does not affect the probability of the collision occurring. Therefore, wearing a helmet does not reduce the likelihood of the hazard event from occurring. It is not a prevention measure.
        
        Mitigation Explanation: Thinking step by step:
        - In the event of a collision, a helmet can absorb some of the impact energy and protect the cyclist's head.
        - By reducing the direct impact to the head, a helmet can lessen the severity of a head injury.
        - While a helmet may not completely eliminate the risk of a head injury, it can significantly reduce the extent of the harm caused. Therefore, wearing a helmet reduces the harm caused by the "Head injury" for the cyclist. It is a mitigation measure.
        
        Answer: Mitigation
        </EXAMPLE OUTPUT>

        <EXAMPLE INSTRUCTIONS>
        1. In one sentence, describe the hazard given the hazard event: "Zip tie projectile hits audience member" during the
        activity: 'Fluids laboratory' given the harm caused: "Impact injury" for Students.
        2. Write the Hazard Event: "Zip tie projectile hits audience member"
        3. Write the Harm Caused: "Impact injury".
        4. Thinking step by step, explain whether or not 'Keeping hand around zip tie when cutting to stop it from flying' reduces the likelihood that hazard event: "Zip tie projectile hits audience member" occurs.
        If so, it is a prevention measure.
        5. Thinking step by step, explain whether or not 'Keeping hand around zip tie when cutting to stop it from flying' removes or reduces the harm caused: "Impact injury" for the 'Students'.
        If so, it is a mitigation measure.
        6. If it is a prevention measure, answer 'Prevention'. If it is a mitigation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a
        prevention measure and a mitigation measure, answer 'Both'.
        </EXAMPLE INSTRUCTIONS>

        <EXAMPLE OUTPUT>
        Hazard Description: During a fluids laboratory, a cut zip tie flies and hits a student audience member, causing an impact injury.
        Hazard Event: Zip tie projectile hits audience member
        Harm Caused: Impact injury

        Prevention Explanation: Thinking step by step:
        - Keeping a hand around the zip tie while cutting it can physically prevent the cut piece from flying away.
        - By containing the cut zip tie, the likelihood of it being launched towards the audience is greatly reduced.
        - Therefore, keeping a hand around the zip tie when cutting it reduces the likelihood of the hazard event from occurring. It is a prevention measure.
        
        Mitigation Explanation: Thinking step by step:
        - If a cut zip tie were to fly and hit a student, keeping a hand around it during cutting would not reduce the impact force.
        - The hand around the zip tie does not provide any protective barrier for the student who might be hit.
        - Therefore, keeping a hand around the zip tie when cutting does not remove or reduce the harm caused by the "Impact injury" for the students. It is not a mitigation measure.
        
        Answer: Prevention
        </EXAMPLE OUTPUT>
        """

        return f'''
        <CONTEXT>
        You are a Risk Assessment expert responsible for giving feedback on Risk Assessment inputs.
        </CONTEXT>

        <STYLE>
        Follow the writing style of a secondary school teacher.
        </STYLE>

        <TONE>
        Use a formal tone in your outputs.
        </TONE>

        <AUDIENCE>
        Your audience is a student who is learning how to write a risk assessment.
        </AUDIENCE>

        {all_few_shot_examples}

        <INSTRUCTIONS>
        Follow these instructions:
        1. In one sentence, describe the hazard given the hazard event: "{hazard_event}" during the
        activity: "{self.activity}" given the harm caused: "{harm_caused}" for {self.who_it_harms}.
        2. Thinking step by step, explain whether or not "{self.control_measure}" reduces the likelihood that the hazard event: "{hazard_event}" occurs.
        If so, it is a prevention measure.
        3. Thinking step by step, explain whether or not "{self.control_measure}" removes or reduces the harm caused: "{harm_caused}" for the "{self.who_it_harms}".
        If so, it is a mitigation measure.
        4. If it is a prevention measure, answer 'Prevention'. If it is a mitigation meausure, answer 'Mitigation'.
        If it is neither a prevention measure nor a mitigation measure, answer 'Neither'. If it is both a
        prevention measure and a mitigation measure, answer 'Both'.
        </INSTRUCTIONS>

        <OUTPUT FORMAT>
        Use the following output format:
        Hazard Description: <your hazard event description>
        Prevention Explanation: <your prevention explanation>
        Mitigation Explanation: <your mitigation explanation>
        Answer: <your answer>
        </OUTPUT FORMAT>
        
        <OUTPUT>
        Hazard Description: '''

class PreventionPrompt(ControlMeasureClassification):
    def __init__(self, control_measure, activity, hazard, how_it_harms, who_it_harms):
        super().__init__(control_measure, activity, hazard, how_it_harms, who_it_harms)
    
    def get_field_checked(self):
        return 'Prevention'
    
    def get_shortform_feedback(self, feedback_type):
        if feedback_type == 'positive':
            return f"Correct! '{self.control_measure}' is a prevention measure for the hazard: '{self.hazard}'"
        if feedback_type == 'both':
            return f"The prevention you entered is both a mitigation and a prevention measure"
        if feedback_type == 'neither':
            return f"Incorrect. '{self.control_measure}' is not a prevention measure for the hazard: '{self.hazard}'."
        if feedback_type == 'misclassification':
            return f"Incorrect. You entered a mitigation measure in the prevention field."
    
    def get_longform_feedback(self, prompt_output, start_string='Prevention Explanation', end_string='Mitigation Explanation'):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.extract_section_of_prompt_between_two_strings(prompt_output=prompt_output, start_string=start_string, end_string=end_string)

    # TODO: When you have hazard event input, can include in feedback.
    def get_recommendation(self, recommendation_type):
        if recommendation_type == 'both':
            return f"""A mitigation measure reduces the harm caused by the hazard event either while the hazard event is occurring or after it has occurred. On the other hand, a prevention measure reduces the likelihood of the hazard event occurring in the first place. Please use the above definitions to check your prevention input."""

        if recommendation_type == 'neither':
            return "For the prevention field, enter a control measure which reduces the likelihood of the hazard event."

        if recommendation_type == 'misclassification':
            return f"""A mitigation measure reduces the harm caused by the hazard. On the other hand, a prevention measure reduces the likelihood of the event that leads to harm. Please use the above definitions to ammend your prevention input."""
    
class MitigationPrompt(ControlMeasureClassification):
    def __init__(self, control_measure, activity, hazard, how_it_harms, who_it_harms):
        super().__init__(control_measure, activity, hazard, how_it_harms, who_it_harms)

    def get_shortform_feedback(self, feedback_type):
        if feedback_type == 'positive':
            return f"Correct! '{self.control_measure}' is a mitigation measure for the hazard: '{self.hazard}'."
        if feedback_type == 'both':
            return f"The mitigation measure you entered is both a prevention and a mitigation."
        if feedback_type == 'neither':
            return f"Incorrect."
        if feedback_type == 'misclassification':
            return f"Incorrect. You entered a prevention measure in the mitigation field."
    
    def get_longform_feedback(self, prompt_output, start_string='Mitigation Explanation', end_string='Answer'):
        regex_pattern_matcher = RegexPatternMatcher()
        return regex_pattern_matcher.extract_section_of_prompt_between_two_strings(prompt_output=prompt_output, start_string=start_string, end_string=end_string)
    
    # TODO: When you have hazard event input, can include in feedback.
    def get_recommendation(self, recommendation_type):
        if recommendation_type == 'both':
            return f"""A prevention measure reduces the likelihood of the hazard event occurring in the first place. On the other hand, a mitigation measure reduces the harm caused by the hazard event while it is happening or after it has occurred. Please use the above definitions to check your mitigation input."""

        if recommendation_type == 'neither':
            return "For the mitigation field, enter a control measure which reduces the harm caused by the hazard event either while the hazard event is occurring or after it has occurred."
        
        if recommendation_type == 'misclassification':
            return f"""A prevention measure reduces the likelihood of the event that leads to harm. On the other hand, a mitigation measure reduces the harm caused by the hazard. Please use the above definitions to ammend your mitigation input."""

class SummarizeControlMeasureFeedback(PromptInput):
    def __init__(self):

        self.candidate_labels = [True, False]
        self.pattern_matching_method = 'always_return_true'

    def get_context_style_tone_audience(self):
        return """<CONTEXT>
        You are a Risk Assessment expert responsible for summarizing feedback on Risk Assessment inputs.
        </CONTEXT>

        <STYLE>
        Follow the writing style of a secondary school teacher.
        </STYLE>

        <TONE>
        Use a formal tone in your outputs.
        </TONE>

        <AUDIENCE>
        Your audience is a student who is learning how to write a risk assessment.
        </AUDIENCE>"""
        
    def get_instructions(self, control_measure_type, feedback):
        return f"""<INSTRUCTIONS>
        Summarize the {control_measure_type} feedback: "{feedback}" in the following format:
        - In 2 sentences, provide an explanation as to why the control measure is or is not a {control_measure_type}. 
        - In one sentence, provide a statement of whether or not control measure is {control_measure_type} measure.
        Start the summary immediately after these instructions.
        </INSTRUCTIONS>"""
    
    def get_example(self, control_measure_type):
        if control_measure_type == 'prevention':
            return f"""<EXAMPLE INPUT>
            Thinking step by step:
            - Enhancing explosive detection capabilities, such as deploying advanced screening equipment or trained explosive detection canine units, increases the chances of identifying and intercepting explosives before they can be detonated.
            - By detecting and preventing explosives from entering the event premises, the likelihood of an explosive terrorist attack occurring is significantly reduced.
            - Therefore, enhancing explosive detection capabilities directly reduces the likelihood of the hazard event "Explosive terrorist attack during public gatherings or events" from occurring. It is a prevention measure.
            </EXAMPLE INPUT>

            <EXAMPLE OUTPUT>
            Enhancing explosive detection capabilities, such as deploying advanced screening equipment or trained explosive detection canine units allows for the detection and prevention of explosives from entering the event premises.
            Thus, the likelihood of an explosive terrorist attack occurring is significantly reduced. Hence, enhancing explosive detection capabilities is a prevention measure.
            </EXAMPLE OUTPUT>"""
        
        if control_measure_type == 'mitigation':
            return '''<EXAMPLE INPUT>
            Thinking step by step:
            - Loud noises can cause hearing damage, especially at close range and with prolonged exposure.
            - By increasing the distance between the trombone player and the audience, the intensity of the sound reaching the audience is reduced.
            - This reduced sound intensity can help reduce hearing damage caused by the loud noise.
            - Therefore, keeping a space between the player and audience removes or reduces the harm caused by "Hearing damage" for everyone present. It is a mitigation measure.
            </EXAMPLE INPUT>

            <EXAMPLE OUTPUT>
            Increasing the distance between the trombone player and the audience reduces the intensity of the sound reaching the audience.
            This helps reduce potential hearing damage from the loud noise. Hence, increasing distance between the trombone player and the audience is a mitigation measure.
            </EXAMPLE OUTPUT>'''

    def generate_prompt(self, control_measure_type, feedback):
        return f'''
        {self.get_context_style_tone_audience()}

        {self.get_example(control_measure_type=control_measure_type)}

        {self.get_instructions(control_measure_type=control_measure_type, feedback=feedback)}'''