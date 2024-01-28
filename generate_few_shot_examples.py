# Current how it harms examples:
from example_risk_assessments import RA_9, RA_4_with_incorrect_how_it_harms

# Current prevention examples:
from example_risk_assessments import RA_4, RA_5, RA_mucking_out_horse, RA_4_with_first_aid

# Current mitigation examples:
from example_risk_assessments import RA_6, RA_mucking_out_horse, RA_4_with_first_aid, RA_21

def get_prevention_prompt(risk_assessment, few_shot=False):
    prevention = risk_assessment.get_prevention_input()
    if few_shot:
        return prevention.generate_prompt()
    else:
        return prevention.generate_prompt_without_few_shot_examples()

def get_mitigation_prompt(risk_assessment, few_shot=False):
    mitigation = risk_assessment.get_mitigation_input()
    if few_shot:
        return mitigation.generate_prompt()
    else:
        return mitigation.generate_prompt_without_few_shot_examples()

def get_how_it_harms_prompt(risk_assessment, few_shot=False):
    how_it_harms = risk_assessment.get_how_it_harms_in_context_input()
    if few_shot:
        return how_it_harms.generate_prompt()
    else:
        return how_it_harms.generate_prompt_without_few_shot_examples()

if __name__ == "__main__":
    # How it harms
    # print(get_how_it_harms_prompt(RA_9)) # Correct example
    # print(get_how_it_harms_prompt(RA_4_with_incorrect_how_it_harms)) # Incorrect example

    # # Prevention
    # print(get_prevention_prompt(RA_5)) # Example of prevention
    # print(get_prevention_prompt(RA_4)) # Example of mitigation ## CAREFUL
    
    # # Mitigation
    # print(get_mitigation_prompt(RA_4_with_first_aid)) # Example where mitigation reduces the harm after hazard event occurred
    print(get_mitigation_prompt(RA_mucking_out_horse)) # Example of mitigation which reduces the harm when the hazard event is occurring.
    # print(get_mitigation_prompt(RA_6)) # Example of prevention which got classified as a mitigation
    # print(get_how_it_harms_prompt(RA_mucking_out_horse, few_shot=True))