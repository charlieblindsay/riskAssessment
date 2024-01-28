from PromptInputs import Mitigation
from ExamplesGenerator import ExamplesGeneratorFromCorrectExamples

class MitigationExamplesGenerator(ExamplesGeneratorFromCorrectExamples):
    def generate_incorrect_example(self, correct_index, incorrect_index):
        return Mitigation(
                mitigation=self.correct_examples_list[incorrect_index].mitigation,
                activity=self.correct_examples_list[correct_index].activity, 
                hazard=self.correct_examples_list[correct_index].hazard, 
                how_it_harms=self.correct_examples_list[correct_index].how_it_harms,
                who_it_harms=self.correct_examples_list[correct_index].who_it_harms)
    
correct_mitigation_examples_list = [
    Mitigation(
        mitigation="Rapid response to contain wildfires",
        activity="Deforestation for agriculture",
        hazard="Wildfires",
        how_it_harms="Habitat destruction",
        who_it_harms="Wildlife"
    ),
    # Mitigation(
    #     mitigation="Installation of containment barriers",
    #     activity="Industrial chemical production and storage",
    #     hazard="Industrial Chemical Spill",
    #     how_it_harms="Environmental contamination",
    #     who_it_harms="Surrounding ecosystems"
    # ),
    # Mitigation(
    #     mitigation="Building elevated structures",
    #     activity="Urbanization in flood-prone areas",
    #     hazard="Flood",
    #     how_it_harms="Property damage",
    #     who_it_harms="Residents in affected areas"
    # ),
    # Mitigation(
    #     mitigation="Implementing emission control technologies",
    #     activity="Heavy reliance on fossil fuels for transportation",
    #     hazard="Urban Air Pollution",
    #     how_it_harms="Respiratory issues",
    #     who_it_harms="Residents in urban areas"
    # ),
    # Mitigation(
    #     mitigation="Establishing quarantine facilities at major transportation hubs",
    #     activity="Global travel without disease prevention measures",
    #     hazard="Epidemic/Pandemic",
    #     how_it_harms="Spread of infectious diseases",
    #     who_it_harms="Travelers"
    # ),
    # Mitigation(
    #     mitigation="Installing slope stabilization measures",
    #     activity="Unregulated construction on hillsides",
    #     hazard="Landslide",
    #     how_it_harms="Structural damage",
    #     who_it_harms="Residents in affected areas"
    # ),
    # Mitigation(
    #     mitigation="Decontamination efforts",
    #     activity="Operating nuclear power plants",
    #     hazard="Nuclear Accident",
    #     how_it_harms="Radioactive contamination",
    #     who_it_harms="Workers at the plant"
    # ),
    # Mitigation(
    #     mitigation="Developing technologies for debris removal",
    #     activity="Poorly managed space debris",
    #     hazard="Space Debris Impact",
    #     how_it_harms="Damage to satellites",
    #     who_it_harms="Space infrastructure"
    # ),
    # Mitigation(
    #     mitigation="Remediation of contaminated sites",
    #     activity="Industrialization without proper waste management",
    #     hazard="Environmental Pollution",
    #     how_it_harms="Health impacts",
    #     who_it_harms="Local communities"
    # )
]