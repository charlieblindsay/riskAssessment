from regex import *
from RegexPatternMatcher import RegexPatternMatcher

# prompt_output = """Explanation 4: 1. Description: The hazard of 'Ink spillage' during the activity 'Fluids laboratory' can lead to serious eye damage to students. 
# 2. Prevention Explanation: 'Wear safety glasses' reduces the likelihood of the hazard of ink spillage leading to serious eye damage. Safety glasses provide a physical barrier that can protect the eyes from direct contact with the ink. Therefore, it is a prevention measure.
# 3. Mitigation Explanation: Assuming the hazard of ink spillage has already led to serious eye damage, 'Wear safety glasses' will not directly reduce or remove the harm caused by the hazard. However, wearing safety glasses can prevent further eye damage and protect the eyes during the recovery process. Therefore, it can be considered a mitigation measure.
# 4. Answer: Both"""

prompt_output = """Description: A fluids laboratory is a controlled environment where experiments and tests are conducted to study the behavior and properties of fluids.

Comparison: The description of a fluids laboratory aligns with the provided definition of an activity as it involves physical and mental effort in conducting experiments and tests.

Answer: True"""

regex_pattern_matcher = RegexPatternMatcher()
print(regex_pattern_matcher.get_explanation_from_prompt_output(prompt_output, 'Comparison', 'Answer'))