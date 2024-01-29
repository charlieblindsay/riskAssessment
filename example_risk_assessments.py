try:
    from .RiskAssessment import RiskAssessment
except ImportError:
    from RiskAssessment import RiskAssessment

RA_empty_input = RiskAssessment(
    activity="",
    hazard="Loud noise",
    who_it_harms="Everyone present",
    how_it_harms="Loud noise from instrument can cause hearing damage.",
    uncontrolled_likelihood="4",
    uncontrolled_severity="1",
    uncontrolled_risk="4",
    prevention="Play quietly, at a volume suitable for the room",
    mitigation="1",
    controlled_likelihood="1",
    controlled_severity="1",
    controlled_risk="1",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='',
)

RA_mitigation_wrong_type = RiskAssessment(
    activity="Using a trombone as a demonstration for a TPS presentation",
    hazard="Loud noise",
    who_it_harms="Everyone present",
    how_it_harms="Loud noise from instrument can cause hearing damage.",
    uncontrolled_likelihood="4",
    uncontrolled_severity="1",
    uncontrolled_risk="4",
    prevention="Play quietly, at a volume suitable for the room",
    mitigation="",
    controlled_likelihood="1",
    controlled_severity="1",
    controlled_risk="1",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='',
)

RA_controlled_likelihood_wrong_type = RiskAssessment(
    activity="Using a trombone as a demonstration for a TPS presentation",
    hazard="Loud noise",
    who_it_harms="Everyone present",
    how_it_harms="Loud noise from instrument can cause hearing damage.",
    uncontrolled_likelihood="4",
    uncontrolled_severity="1",
    uncontrolled_risk="4",
    prevention="Play quietly, at a volume suitable for the room",
    mitigation="Keep a space between the player and audience",
    controlled_likelihood="Keep a space between the player and audience",
    controlled_severity="1",
    controlled_risk="1",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
)

RA_incorrect_prevention_and_mitigation = RiskAssessment(
    activity="Welding metal structures",
    hazard="Exposure to toxic welding fumes",
    how_it_harms="Inhaling welding fumes can lead to respiratory problems, lung damage, and long-term health issues.",
    who_it_harms="Welders and individuals in the vicinity of the welding area.",
    uncontrolled_likelihood="4",
    uncontrolled_severity="3",
    uncontrolled_risk="12",
    prevention="Using the welding equipment in an enclosed space without proper ventilation.",
    mitigation='',
    controlled_likelihood="1",
    controlled_severity="1",
    controlled_risk="1",
    prevention_prompt_expected_output='neither',
    mitigation_prompt_expected_output='neither',

)

RA_1 = RiskAssessment(
    activity="Using a trombone as a demonstration for a TPS presentation",
    hazard="Impact from instrument",
    who_it_harms="Audience",
    how_it_harms="Slide could hit audience member, causing impact injury.",
    uncontrolled_likelihood="4",
    uncontrolled_severity="2",
    uncontrolled_risk="8",
    prevention="Keep safe distance between the player and audience; hold instrument securely",
    mitigation="",
    controlled_likelihood="1",
    controlled_severity="2",
    controlled_risk="2",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='',
)

RA_2_hearing_damage = RiskAssessment(
    activity="Using a trombone as a demonstration for a TPS presentation",
    hazard="Loud noise",
    who_it_harms="Everyone present",
    how_it_harms="Loud noise from instrument can cause hearing damage.",
    uncontrolled_likelihood="4",
    uncontrolled_severity="1",
    uncontrolled_risk="4",
    prevention="Play quietly, at a volume suitable for the room", # reduces likelihood of loud noise
    mitigation="Keep a space between the player and audience", # reduces severity of loud noise
    controlled_likelihood="1",
    controlled_severity="1",
    controlled_risk="1",
    prevention_prompt_expected_output='prevention', 
    mitigation_prompt_expected_output='mitigation',
)

RA_2_mitigation_prevention_switched = RiskAssessment(
    activity="Using a trombone as a demonstration for a TPS presentation",
    hazard="Loud noise",
    who_it_harms="Everyone present",
    how_it_harms="Loud noise from instrument can cause hearing damage.",
    uncontrolled_likelihood="4",
    uncontrolled_severity="1",
    uncontrolled_risk="4",
    mitigation="Play quietly, at a volume suitable for the room", # reduces likelihood of loud noise
    prevention="Keep a space between the player and audience", # reduces severity of loud noise
    controlled_likelihood="1",
    controlled_severity="1",
    controlled_risk="1",
    prevention_prompt_expected_output='mitigation', 
    mitigation_prompt_expected_output='prevention',
)


RA_3_water_from_instrument = RiskAssessment(
    activity="Using a trombone as a demonstration for a TPS presentation",
    hazard="Water from instrument",
    who_it_harms="Audience",
    how_it_harms="Condensation formed in instrument could spread germs if released",
    uncontrolled_likelihood="4",
    uncontrolled_severity="1",
    uncontrolled_risk="4",
    prevention="Ensure water is not released during presentation", # Not very specific.
    # Should include feedback stating: "How would you ensure water is not released during presentation?"
    mitigation="Keep a space between the player and audience", # Reduces severity of water being released
    controlled_likelihood="1",
    controlled_severity="1",
    controlled_risk="1",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='prevention', # reduces likelihood that someone becomes ill
)

RA_3_water_from_instrument_mitiagation_prevention_switched = RiskAssessment(
    activity="Using a trombone as a demonstration for a TPS presentation",
    hazard="Water from instrument",
    who_it_harms="Audience",
    how_it_harms="Condensation formed in instrument could spread germs if released",
    uncontrolled_likelihood="4",
    uncontrolled_severity="1",
    uncontrolled_risk="4",
    mitigation="Ensure water is not released during presentation", # Not very specific.
    # Should include feedback stating: "How would you ensure water is not released during presentation?"
    prevention="Keep a space between the player and audience", # Reduces severity of water being released
    controlled_likelihood="1",
    controlled_severity="1",
    controlled_risk="1",
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='prevention',
)

RA_4 = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Ink spillage",
    who_it_harms="Students",
    how_it_harms="Serious eye damage",
    uncontrolled_likelihood="2",
    uncontrolled_severity="3",
    uncontrolled_risk="6",
    prevention="Wear safety glasses", # reduces likelihood of ink getting in your eyes
    mitigation="Wash your eyes with clean water", # reduces severity after hazard has occurred
    controlled_likelihood="1",
    controlled_severity="3",
    controlled_risk="3",
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='mitigation',
)

RA_4_with_first_aid = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Ink spillage",
    who_it_harms="Students",
    how_it_harms="Serious eye damage",
    uncontrolled_likelihood="2",
    uncontrolled_severity="3",
    uncontrolled_risk="6",
    prevention="Wear safety glasses", # reduces likelihood of hazard occurring
    mitigation="First aid", # reduces severity after hazard has occurred
    controlled_likelihood="1",
    controlled_severity="3",
    controlled_risk="3",
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='mitigation',
)


RA_4_with_incorrect_how_it_harms = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Ink spillage",
    who_it_harms="Students",
    how_it_harms="Radiation exposure",
    uncontrolled_likelihood="2",
    uncontrolled_severity="3",
    uncontrolled_risk="6",
    prevention="Wear safety glasses", # reduces likelihood of hazard occurring
    mitigation="Wash your eyes with clean water", # reduces severity after hazard has occurred
    controlled_likelihood="1",
    controlled_severity="3",
    controlled_risk="3",
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='mitigation',
)

RA_5 = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Wet hands",
    who_it_harms="Students",
    how_it_harms="Electric shock of students when touching electronics (pump power supply) with wet hands",
    uncontrolled_likelihood="3",
    uncontrolled_severity="3",
    uncontrolled_risk="9",
    prevention="Students should make sure they touch electronics only with dry hands", # reduces likelihood of hazard occurring
    mitigation="Unplug the pump and call for urgent medical assistance", # reduces severity after hazard has occurred
    controlled_likelihood="2",
    controlled_severity="3",
    controlled_risk="6",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
)

RA_5_mitigation_prevention_switched = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Wet hands",
    who_it_harms="Students",
    how_it_harms="Electric shock of students when touching electronics (pump power supply) with wet hands",
    uncontrolled_likelihood="3",
    uncontrolled_severity="3",
    uncontrolled_risk="9",
    mitigation="Students should make sure they touch electronics only with dry hands", # reduces likelihood of hazard occurring
    prevention="Unplug the pump and call for urgent medical assistance", # reduces severity after hazard has occurred
    controlled_likelihood="2",
    controlled_severity="3",
    controlled_risk="6",
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='prevention',
)

RA_6 = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Tripping over personal belongings",
    who_it_harms="Students",
    how_it_harms="Tripping can cause physical harm.",
    uncontrolled_likelihood="5",
    uncontrolled_severity="2",
    uncontrolled_risk="10",
    prevention="Put all belongings away from footpaths", # This reduces likelihood of hazard occurring and is therefore a prevention measure
    mitigation="Take care when walking around", # This reduces likelihood of hazard occurring and is therefore a prevention measure 
    controlled_likelihood="1",
    controlled_severity="2",
    controlled_risk="2",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='prevention',
)

RA_7_water_tank = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Water being spilt on the floor",
    who_it_harms="Students",
    how_it_harms="Injuries caused by possible slipping on wet floor",
    uncontrolled_likelihood="3",
    uncontrolled_severity="2",
    uncontrolled_risk="6",
    prevention="Do not move the water tank when it is full",
    mitigation="""If someone gets injured due to slipping, apply an ice pack to the injured area and 
    seek medical advice without delay.""",
    controlled_likelihood="2",
    controlled_severity="2",
    controlled_risk="4",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
)

RA_7_water_tank_mitigation_prevention_switched = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Water being spilt on the floor",
    who_it_harms="Students",
    how_it_harms="Injuries caused by possible slipping on wet floor",
    uncontrolled_likelihood="3",
    uncontrolled_severity="2",
    uncontrolled_risk="6",
    mitigation="Do not move the water tank when it is full",
    prevention="""If someone gets injured due to slipping, apply an ice pack to the injured area and 
    seek medical advice without delay.""",
    controlled_likelihood="2",
    controlled_severity="2",
    controlled_risk="4",
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='prevention',
)


RA_8_syringe_needle = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Syringes with sharp needles",
    who_it_harms="Students",
    how_it_harms="Sharp needles can pierce the skin and cause bleeding",
    uncontrolled_likelihood="3",
    uncontrolled_severity="3",
    uncontrolled_risk="9",
    prevention="Point needle away from yourself and others",
    mitigation="Wear lab coat and PPE", # This is both prevention and mitigation
    controlled_likelihood="2",
    controlled_severity="1",
    controlled_risk="2",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
)

RA_8_syringe_needle_mitigation_prevention_switched = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Syringes with sharp needles",
    who_it_harms="Students",
    how_it_harms="Sharp needles can pierce the skin and cause bleeding",
    uncontrolled_likelihood="3",
    uncontrolled_severity="3",
    uncontrolled_risk="9",
    mitigation="Point needle away from yourself and others",
    prevention="Wear lab coat and PPE", # This is both prevention and mitigation
    controlled_likelihood="2",
    controlled_severity="1",
    controlled_risk="2",
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='prevention',
)

RA_9 = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Electrocution",
    who_it_harms="Students",
    how_it_harms="Electrocuted by mains voltage", # This is a description of the process not the harm on the students
    uncontrolled_likelihood="2",
    uncontrolled_severity="2",
    uncontrolled_risk="4",
    prevention="Pump plug stays away from water",
    mitigation="First aid on site", # There needs to be a description of how this will be used in the event of an electrocution.
    controlled_likelihood="1",
    controlled_severity="2",
    controlled_risk="2",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
)

RA_9_mitigation_prevention_switched = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Electrocution",
    who_it_harms="Students",
    how_it_harms="Electrocuted by mains voltage", # This is a description of the process not the harm on the students
    uncontrolled_likelihood="2",
    uncontrolled_severity="2",
    uncontrolled_risk="4",
    mitigation="Pump plug stays away from water",
    prevention="First aid on site", # There needs to be a description of how this will be used in the event of an electrocution.
    controlled_likelihood="1",
    controlled_severity="2",
    controlled_risk="2",
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='prevention',
)

RA_13 = RiskAssessment(
    activity='Presentation Demonstration',
    hazard='Demonstration with ruler, styrofoam and bbq sticks. I will be flicking the ruler while clamping it to a table. The bbq sticks will be stuck in the styrofoam and I will shake to show resonance',
    who_it_harms='Me and audience',
    how_it_harms='Could hit someone or the demonstration falls apart',
    uncontrolled_likelihood='4',
    uncontrolled_severity='1',
    uncontrolled_risk='4',
    prevention='Do the demonstration with care and keep space from audience',
    mitigation='',
    controlled_likelihood='2',
    controlled_severity='1',
    controlled_risk='2',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='',
)

RA_14 = RiskAssessment(
    activity='Using  paper plane models as a demonstration for a TPS presentation',
    hazard='Impact from instuments',
    who_it_harms='Audience',
    how_it_harms='Plane could hit audience member, causing impact injury.',
    uncontrolled_likelihood='4',
    uncontrolled_severity='2',
    uncontrolled_risk='8',
    prevention='Keep safe distance between the player and audience; Throw the paper plane to a direction without anyone',
    mitigation='',
    controlled_likelihood='1',
    controlled_severity='2',
    controlled_risk='2',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='',
)

RA_15 = RiskAssessment(
    activity='TPS presentation',
    hazard='Climbing Protection Gear (Cams and Hexs)',
    who_it_harms="Students and other individuals who would like to see how they work.",
    how_it_harms="""Some equipment is heavy so could hurt if dropped on feet.""",
    uncontrolled_likelihood='2',
    uncontrolled_severity='1',
    uncontrolled_risk='2',
    prevention='Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly.',
    mitigation='First aid if necessary',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
)

RA_15_mitigation_prevention_switched = RiskAssessment(
    activity='TPS presentation',
    hazard='Climbing Protection Gear (Cams and Hexs)',
    who_it_harms="Students and other individuals who would like to see how they work.",
    how_it_harms="""Some equipment is heavy so could hurt if dropped on feet.""",
    uncontrolled_likelihood='2',
    uncontrolled_severity='1',
    uncontrolled_risk='2',
    mitigation='Inform those who wish to hold the equipment of the risk and demonstrate how they are used correctly.',
    prevention='First aid if necessary',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='prevention',
)

RA_17 = RiskAssessment(
    activity='Bringing in a drone and two empty Li-Po batteries',
    hazard='Sharp Edge of propellor blade on drone',
    who_it_harms='Whoever pokes the propellor blade at the tip',
    how_it_harms='Is sharp to the touch to cause pain but not sharp enough to pierce skin',
    uncontrolled_likelihood='1',
    uncontrolled_severity='1',
    uncontrolled_risk='1',
    prevention='Make them aware the tip is sharp',
    mitigation='',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='',
)

RA_18 = RiskAssessment(
    activity='Bringing in a drone and two empty Li-Po batteries',
    hazard='Li-Po battery to handle',
    who_it_harms='Whoever is holding it',
    how_it_harms='It may heat up with an unlikely chance of a fire',
    uncontrolled_likelihood='1',
    uncontrolled_severity='2',
    uncontrolled_risk='2',
    prevention='Li-Po batteries have been discharged to a safe level and are being carried in a fire-proof bag',
    mitigation='Don\'t let the audience handle it for too long',
    controlled_likelihood='1',
    controlled_severity='2',
    controlled_risk='2',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='prevention', # it is both mitigation and prevention
)

RA_19 = RiskAssessment(
    activity='Using a spring contraption as a demonstration for a TPS presentation',
    hazard='Slippage of weight for contraption',
    who_it_harms='Demonstrator',
    how_it_harms='Heavy impact when falling onto demonstator, causing injury',
    uncontrolled_likelihood='3',
    uncontrolled_severity='4',
    uncontrolled_risk='12',
    prevention='Make sure the mass is properly secured to the contraption and the contraption is secured.',
    mitigation='Keep away from below the contraption',
    controlled_likelihood='1',
    controlled_severity='2',
    controlled_risk='2',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='prevention',
)

RA_20 = RiskAssessment(
    activity='Using a spring contraption as a demonstration for a TPS presentation',
    hazard='Cut Zip tie may fly',
    who_it_harms='Audience',
    how_it_harms='When cut the zip tie may hit an audience member',
    uncontrolled_likelihood='4',
    uncontrolled_severity='2',
    uncontrolled_risk='8',
    prevention='Keep hand around zip tie when cutting to stop it from flying',
    mitigation='Ensure safe distance between contraption and audience.',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='prevention', # Another prevention measure as it reduces the likelihood of the zip tie hitting an audience member
)

RA_23 = RiskAssessment(
    activity='Using a mechanical pencil and breaking the pencil lead against a surface for demonstration',
    hazard='Pencil lead breaking and becoming a projectile',
    who_it_harms='Anyone present',
    how_it_harms='May enter one\'s eye',
    uncontrolled_likelihood='3',
    uncontrolled_severity='2',
    uncontrolled_risk='6',
    prevention='Keep safe distance between the audience when demonstrating lead breakage',
    mitigation='',
    controlled_likelihood='2',
    controlled_severity='1',
    controlled_risk='2',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='',
)

RA_mucking_out_horse = RiskAssessment(
    activity='Mucking out a horse',
    hazard='Horse kicks out',
    who_it_harms='Horse rider',
    how_it_harms='Impact injury',
    uncontrolled_likelihood='4',
    uncontrolled_severity='3',
    uncontrolled_risk='12',
    prevention='Keep a safe distance from the horse',
    mitigation='Wear a helmet and body protector',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation'
)

RA_mucking_out_horse_mitigation_prevention_switched = RiskAssessment(
    activity='Mucking out a horse',
    hazard='Horse kicks out',
    who_it_harms='Horse rider',
    how_it_harms='Impact injury',
    uncontrolled_likelihood='4',
    uncontrolled_severity='3',
    uncontrolled_risk='12',
    mitigation='Keep a safe distance from the horse',
    prevention='Wear a helmet and body protector',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='mitigation',
    mitigation_prompt_expected_output='prevention'
)


example_risk_assessments = [RA_1, RA_2_hearing_damage, RA_3_water_from_instrument, 
                            RA_4, RA_5, RA_6, RA_7_water_tank, RA_8_syringe_needle, RA_9,
                            RA_13, RA_14, RA_15, RA_17, RA_18, RA_19, RA_20,
                            RA_23, RA_23,
                            RA_incorrect_prevention_and_mitigation, RA_2_mitigation_prevention_switched,
                            RA_3_water_from_instrument_mitiagation_prevention_switched, RA_4_with_first_aid,
                            RA_5_mitigation_prevention_switched, RA_7_water_tank_mitigation_prevention_switched,
                            RA_8_syringe_needle_mitigation_prevention_switched, RA_9_mitigation_prevention_switched,
                            RA_15_mitigation_prevention_switched,
                            RA_mucking_out_horse]

print(len(example_risk_assessments))