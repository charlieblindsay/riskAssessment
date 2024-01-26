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

RA_2 = RiskAssessment(
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

RA_3 = RiskAssessment(
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
    mitigation_prompt_expected_output='mitigation',
)

RA_4 = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Ink spillage",
    who_it_harms="Students",
    how_it_harms="Serious eye damage",
    uncontrolled_likelihood="2",
    uncontrolled_severity="3",
    uncontrolled_risk="6",
    prevention="Wear safety glasses", # reduces likelihood of hazard occurring
    mitigation="Wash your eyes with clean water", # reduces severity after hazard has occurred
    controlled_likelihood="1",
    controlled_severity="3",
    controlled_risk="3",
    prevention_prompt_expected_output='prevention',
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
    prevention_prompt_expected_output='prevention',
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

RA_7 = RiskAssessment(
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

RA_8 = RiskAssessment(
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

RA_10 = RiskAssessment(
    activity="Fluids laboratory",
    hazard="High speed air in wind tunnels",
    who_it_harms="Students",
    how_it_harms="Impact injury",
    uncontrolled_likelihood="4",
    uncontrolled_severity="3",
    uncontrolled_risk="12",
    prevention="Stay out of flow of wind tunnel",
    mitigation="Wear lab coat and PPE",
    controlled_likelihood="1",
    controlled_severity="2",
    controlled_risk="2",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
)

RA_11 = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Broken shards of glass",
    who_it_harms="Students",
    how_it_harms="Get trapped in soles of shoes",
    uncontrolled_likelihood="3",
    uncontrolled_severity="3",
    uncontrolled_risk="9",
    prevention="Handle equipment with care",
    mitigation="Vacate area of damage",
    # Above mitigation is a tricky one. It would reduce severity if student doesn't
    # already have shards of glass in their shoes. But if they do, it would not help.
    controlled_likelihood="2",
    controlled_severity="1",
    controlled_risk="2",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='neither',
)

RA_12 = RiskAssessment(
    activity="Fluids laboratory",
    hazard="Wires/tubing",
    who_it_harms="Anyone present", # Not specific
    how_it_harms="Trip/entanglement",
    uncontrolled_likelihood="2",
    uncontrolled_severity="4",
    uncontrolled_risk="8",
    prevention="Keep a tidy workspace",
    mitigation="First aid on site",
    controlled_likelihood="1",
    controlled_severity="3",
    controlled_risk="3",
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
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

RA_16 = RiskAssessment(
    activity='TPS presentation: Can a human break a diving board?',
    hazard='Loud noise',
    who_it_harms='Everyone present',
    how_it_harms='Loud microphone volume could cause hearing damage',
    uncontrolled_likelihood='4',
    uncontrolled_severity='1',
    uncontrolled_risk='4',
    prevention='Check microphone volume before beginning presentation, ensure volume is sufficiently but not excessively loud',
    mitigation='Keep space between audience and speaker',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='mitigation',
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
    mitigation_prompt_expected_output='both', # it is both mitigation and prevention
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
    mitigation_prompt_expected_output='mitigation',
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
    mitigation_prompt_expected_output='mitigation', # Another prevention measure as it reduces the likelihood of the zip tie hitting an audience member
)

RA_21 = RiskAssessment(
    activity='Showing an artificial snowman for a TPS presentation',
    hazard='Allergies',
    who_it_harms='Audience',
    how_it_harms='Contact with the artificial snow (polymers) might cause allergies',
    uncontrolled_likelihood='4',
    uncontrolled_severity='2',
    uncontrolled_risk='8',
    prevention='Keep safe distance between the snowman and audience',
    mitigation='Ask about allergies beforehand',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='prevention',
    # "Ask about allergies beforehand" is a prevention measure for the hazard of allergies 
    # during the activity of showing an artificial snowman for a TPS presentation. This is 
    # because by asking about allergies beforehand, the presenter can identify individuals who 
    # may be allergic to the artificial snow and take appropriate measures to prevent contact 
    # with the allergen.
)

RA_22 = RiskAssessment(
    activity='Showing an artificial snowman for a TPS presentation',
    hazard='Poison',
    who_it_harms='Audience',
    how_it_harms='Eat the demonstration aritificial snowman',
    uncontrolled_likelihood='4',
    uncontrolled_severity='2',
    uncontrolled_risk='8',
    prevention='Keep safe distance between the snowman and audience',
    mitigation='Warn them it is fake snow, don’t try to taste it',
    controlled_likelihood='1',
    controlled_severity='1',
    controlled_risk='1',
    prevention_prompt_expected_output='prevention',
    mitigation_prompt_expected_output='prevention', # Another prevention measure,
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


example_risk_assessments = [RA_1, RA_2, RA_3, RA_4, RA_5, RA_6, RA_7, RA_8, RA_9, RA_10, RA_11, 
                            RA_12, RA_13, RA_14, RA_15, RA_16, RA_17, RA_18, RA_19, RA_20, RA_21,
                            RA_22, RA_23, RA_23, RA_incorrect_prevention_and_mitigation]