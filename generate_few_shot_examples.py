from PromptInputs import PromptInput, Prevention, Mitigation, HowItHarmsInContext, WhoItHarmsInContext

# For prevention/mitigation
from example_risk_assessments import RA_4, RA_5, RA_incorrect_prevention_and_mitigation

# for how it harms:
from example_risk_assessments import RA_4_with_incorrect_how_it_harms

def get_prevention_prompt(risk_assessment):
    prevention = risk_assessment.get_prevention_input()

    return prevention.generate_prompt()

def get_mitigation_prompt(risk_assessment):
    mitigation = risk_assessment.get_mitigation_input()

    return mitigation.generate_prompt()

def get_how_it_harms_prompt(risk_assessment):
    how_it_harms = risk_assessment.get_how_it_harms_in_context_input()

    return how_it_harms.generate_prompt()

if __name__ == "__main__":
    # print(get_mitigation_prompt(RA_4))
    # print(get_prevention_prompt(RA_5))

    print(get_how_it_harms_prompt(RA_4_with_incorrect_how_it_harms))
    # print(get_prevention_prompt(RA_incorrect_prevention_and_mitigation))