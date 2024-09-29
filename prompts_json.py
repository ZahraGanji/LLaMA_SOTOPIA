from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
torch.cuda.empty_cache()

# Step 5: Define JSON schema


def llama_3_scene_prompt_creation(interaction_type):
    print("in llama_3_scene_prompt_creation")
    json_format = {
        "name": "Name of the movie",
        "description": "A description of the movie scene and interactions and why this scene is chosen for the given interaction type.",
        "character1": "the first main characters involved in the movie scene.",
        "character2": "the second main characters involved in the movie scene.",
        "setting": "Time and place where the scene occurs, including the physical environment and cultural context. This helps establish the mood and tone, influencing the story and characters.",
        "goals1": "The goals and motivations of character1 involved in the scene",
        "goals2": "The goals and motivations of character2 involved in the scene"
    }


    system_message = f"""
    #### Persona: ###
    Imagine you are a movie expert with an exceptional memory for movie scenes.
    Your deep understanding of human social interactions allows you to categorize them into five key
    types: exchange, competition, cooperation, conflict, and coercion.
    Here is a consice explanation about each of the categories for your guidance:
    Exchange: A social interaction where parties trade goods, services, or favors, with mutual benefit.
    Competition: Individuals or groups vie against each other for resources, status, or success.
    Cooperation: Parties work together toward a common goal, sharing effort and benefits.
    Conflict: Opposing interests clash, leading to disagreement or confrontation between individuals or groups.
    Coercion: One party forces or manipulates another into action or agreement, often using power or threats.
    ### Goal: ###
    When the user requests an example of a movie scene from any of these human social interaction categories, provide a detailed description of a relevant movie scene,
    including:
    - Name of the movie
    - A vivid description of how this scene demonstrates the given interaction type.
    - the first main characters involved in the movie scene.
    - the second main characters involved in the movie scene.
    - The setting of the movie scene. Time and place where the scene occurs, including the physical environment and cultural context. This helps establish the mood and tone, influencing the story and characters.
    - The goals and motivations of character1 involved in the scene.
    - The goals and motivations of character2 involved in the scene.
    """

    # TODO make a good prompt here
    user_message = f"""
    ### Question: ###
    Please find a movie scene that demonstrates the {interaction_type} type of human social interaction between 2 agents.
    Your response should include a vivid scene description, aligned with the provided instructions.

    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    return(messages)

def llama_3_scenario_prompt_creation(movie_scene, movie_name, interaction_type, character1,character2, goals1, goals2, setting):
    print("in llama_3_scenario_prompt_creation")

    json_format = {
        "title": "Title of the scenario.",
        "description": "A vivid description of the scenario.",
        "setting": "Description of the time, place, and context of the scenario.",
        "goal1": "The social goals of first agent involved in the scenario base on the goals of the characters of the given movie scene. When creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.",
        "goal2": "The social goals of second agent involved in the scenario base on the goals of the characters of the given movie scene. When creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.",
        "interaction": "Provide a detailed description of how the agents interact with one another in the scenario, provide more than one word."
        }


    system_message = f"""
    #### Persona: ###
    Imagine you're a film expert with an exceptional memory for movie scenes and a deep understanding of human social interactions. You categorize these interactions into five key types: exchange, competition, cooperation, conflict, and coercion.
    For your reference, here is a brief explanation of each category:
    Exchange: A social interaction where parties trade goods, services, or favors for mutual benefit.
    Competition: Individuals or groups strive against one another for resources, status, or success.
    Cooperation: Parties collaborate toward a shared goal, pooling efforts and distributing the rewards.
    Conflict: Opposing interests lead to disagreement or confrontation between individuals or groups.
    Coercion: One party forces or manipulates another into taking action or agreeing to something, often through power or threats.

    ### Goal: ###
    When a user requests a scenario based on a movie scene and a specific type of interaction, create a vivid and detailed human interaction scenario that reflects both the given movie and the chosen interaction type.
    The scenario should depict a meaningful social interaction between two agents (Agent 1 and Agent 2), highlighting their relationship—whether they are strangers, acquaintances, friends, romantic partners, or family members—and the purpose behind each agent’s interaction with the other.
    Please use "agent" instead of specific names in the result scenario and keep all the mentions gender-neutral.
    Include the following details in your scenario description:
    - Title: give a title to the generated scenario.
    - description: A vivid description of the scenario.
    - Setting: Describe the time, place, and context of the scenario.
    - The social goals of first agent involved in the scenario base on the goals of the characters of the given movie scene. When creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
    - The social goals of second agent involved in the scenario base on the goals of the characters of the given movie scene. When creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
    - Interaction: Provide a detailed description of how the agents interact with one another in the scenario.
    """

    user_message = f"""
    ### Question: ###
    Based on the movie scene {movie_scene} from {movie_name}, which showcases a {interaction_type} type of human interaction, featuring the characters {character1} and {character2} with their respective goals and motivations {goals1}, {goals2}, and taking place in the setting {setting}, please generate a detailed description of the human interaction scenario.
    Your response should include a vivid scenario description, aligned with the provided instructions.

    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    return(messages)

def llama_3_episode_prompt_creation(movie1,movie2, interaction_type,scenario_name,scenario, setting, interaction, agent1, agent2, goals1,goals2):
    print("in llama_3_episode_prompt_creation")
    json_format = {
        "Agent A": ["List of the first agents actions with respect to the order of interactions."],
        "Agent B": ["List of the second agents actions with respect to the order of interactions."],
        }

    system_message = f"""
    #### Persona: ###
    Imagine you are a film expert with an exceptional memory for iconic scenes and a deep understanding of human social dynamics. You categorize social interactions into five key types: exchange, competition, cooperation, conflict, and coercion.

    Below is a concise explanation of each interaction type:

    Exchange: A mutually beneficial interaction where parties trade goods, services, or favors.
    Competition: Individuals or groups vie for resources, status, or success, often at the expense of others.
    Cooperation: Parties work together toward a shared goal, combining their efforts for mutual benefit.
    Conflict: Opposing interests clash, leading to disagreement or confrontation between individuals or groups.
    Coercion: One party forces or manipulates another into action or agreement, typically through power or threats.


    ### Goal: ###

    When a user introduces two characters from a movie, you should situate them within the given scenario and setting, 
    stimulating their interaction and roleplay them based on their characteristics and traits—such as age, gender, moral values, and secrets from the given movie (leveraging your expertise in cinema). 
    Additionally, consider their social objectives as provided.

    In this simulation, characters are referred to as agents, and they take turns interacting in a round-robin fashion (e.g., Agent 1 acts first, followed by Agent 2, and so on). On their turn, each agent can choose one of the following actions:

    Speak (verbal communication),
    Use non-verbal communication (e.g., smiling, hugging),
    Take a physical action (e.g., playing music).
    These are key components of social interaction, but agents may also opt to do nothing (remaining silent) or decide to leave, thereby ending the episode.

    Each interaction is limited to a maximum of 20 turns, as most tasks can typically be resolved within this range. The episode concludes when one agent chooses to leave or the turn limit is reached.

    The output must follow the strict JSON format with two lists of ordered actions as follows:
    - Agent A: List of the first agent's actions in the correct order.
    - Agent B: List of the second agent's actions in the correct order.

    """

    user_message = f"""
    ### Question: ###
    Character 1: {agent1} from {movie1} 
    Character 2: {agent2} from {movie2} 
    Scenario: {scenario_name}:{scenario} set in {setting}, showcasing the interaction: {interaction} between the characters, which exemplifies the {interaction_type} type of human interaction.
    character1's objectives in the scenario:{goals1}
    character2's objectives in the scenario:{goals2}
    Please simulate their interaction in two ordered lists as outlined in your instructions.

    Your output should align with your instructions.
    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    return(messages)

# 'interaction type here?
def llama_3_evaluation_prompt_creation(agentA_actions, agentB_actions, agent1, agent2, movie1, movie2,scenario, goals1,goals2):
    print("in llama_3_evaluation_prompt_creation")
    json_format = {
        "Agent A": {
            "Believability": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Relationship": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Knowledge": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Secret": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Social Rules": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Financial and Material Benefits": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Goal": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
        },
        "Agent B": {
            "Believability": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Relationship": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Knowledge": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Secret": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Social Rules": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Financial and Material Benefits": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Goal": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
        }
    }

    system_message = f"""
    #### Persona: ###
    Imagine you are a film expert with an exceptional memory for iconic scenes and a deep understanding of human social dynamics. You categorize social interactions into five key types: exchange, competition, cooperation, conflict, and coercion.
    Below is a concise explanation of each interaction type:
    Exchange: A mutually beneficial interaction where parties trade goods, services, or favors.
    Competition: Individuals or groups vie for resources, status, or success, often at the expense of others.
    Cooperation: Parties work together toward a shared goal, combining their efforts for mutual benefit.
    Conflict: Opposing interests clash, leading to disagreement or confrontation between individuals or groups.
    Coercion: One party forces or manipulates another into action or agreement, typically through power or threats.

    You can evaluate the interaction between two agents based on 7 distinct dimensions, providing a score for each dimension within the range [lower bound–upper bound] which specified for each dimension in the description bellow.
    Below is a detailed explanation of each dimension:

    Goal Completion (G OAL) [0–10] is the extent to which the agent achieved their goals. Agents’ social goals, defined by the environment, are the primary drivers of their behavior.
    Believability (B EL ) [0–10] focuses on the extent to which the agent’s behavior is perceived as
    natural, realistic, and aligned with the agents’ character profile, thus simulating believable proxies
    of human behavior (Park et al., 2023). Specifically, we consider the following criteria: 1. If the
    agent interacts with others in a natural and realistic manner (naturalness). 2. If the actions of the
    agent align with their character traits e.g., personality, values, etc. (consistency).
    Knowledge (K NO) [0–10] captures the agent’s ability to actively acquire new information. This
    dimension is motivated by the fact that curiosity, i.e., the desire to desire to know or learn, is a fun-
    damental human trait (Reiss, 2004; Maslow, 1943). Specifically, we consider the following criteria:
    What information the agent has gained through the interaction, whether the information the agent
    has gained is new to them, and whether the information the agent has gained is important to them.
    Secret (S EC) [-10-0]3 measures the need for agents (humans) to keep their secretive information or
    intention private (Reiss, 2004). From a game theory perspective, leaking secrets often leads to a loss
    of utility (Gilpin & Sandholm, 2006). However, revealing secrets can be a powerful tool to build
    trust and thus improve relationships (Jaffé & Douneva, 2020). In this dimension, we ask what secret
    or secretive intention the participant wants to keep, and whether they keep it successfully.
    Relationship (REL) [-5–5] captures the fundamental human need for social connection and be-
    longing (Maslow, 1943; Bénabou & Tirole, 2006). In this dimension, we ask what relationship
    the participant has with the other agent(s) before the interaction, and then evaluate if the agents’
    interactions with others help preserve or enhance their personal relationships. Additionally, we
    ascertain whether these interactions also impact the social status or the reputation of the agent.
    Social Rules (SOC) [-10–0] concerns norms, regulations, institutional arrangements, and rituals. We
    differentiate between two types of social rules: social norms and legal rules. Legal rules encompass
    prohibited actions and the potential for punishment by institutionalized force, while social norms
    encompass normative social rules (e.g., it is considered rude to speak loudly in a library).
    Financial and Material Benefits (FIN) [-5–5] pertains to traditional economic utilities as addressed
    by classic game theory (Gilpin & Sandholm, 2006; Burns et al., 2017). We consider financial util-
    ity to be comprised of both short-term monetary benefits (e.g., earnings) and long-term economic
    payoffs (e.g., job security, stock holdings, funding opportunities).

    ### Goal: ###

    When a user presents a simulated interaction between two movie characters, structured as two ordered lists of actions within a defined scenario with specified character goals, your task is to evaluate the interaction. Using the provided instructions and your knowledge of the characters' personalities and behaviors from the films, assess the interaction across the following seven dimensions, assigning a score within the specified range for each.
    BEL
    Reasoning requirement: 1. Evaluate if the agent interacts with
    others in a natural and realistic manner (here are a few common
    questions to check: a. whether the agent is confusing with its own
    identity? b. whether the agent repeats others’ words/actions
    without any reason?c. whether the agent is being overly
    polite considering the context?). Start the analysis with tag
    <naturalness> 2. Analyze whether the actions of the agent align
    with their character traits (e.g., personality, values, and etc.).
    Start the analysis with tag <consistency>. Output your reasoning
    process to the ‘reasoning’ field. Output an integer score ranging
    from 0 and 10 in the ’score’ field. A higher score indicates that
    the agent is more believable.

    REL
    Please first analyze what relationship the participant has with
    the other agent(s) before the interaction. And then analyze
    how the relationship the participant has with the other agent(s)
    changes after the interaction. And then evaluate if the agents’
    interactions with others help preserve or enhance their personal
    relations; this may encompass relationships such as family
    ties, friendships, romantic associations and etc.
    Additionally, ascertain whether these interactions also impact their social
    status or reputation.
    In the ‘reasoning’ field, provide a
    comprehensive account of the logic or thought process that led
    you to your conclusion. Further, provide an integer score ranging
    from -5 to 5 in the ‘score’ field. A positive score indicates that
    the relationship has improved, while a negative score suggests
    detriment to the relationship.
    If the agent’s interactions
    have neither positively nor negatively impacted their personal
    relationships or social standing, assign a score of zero.

    KNO
    Please first assess what information the agent has gained through
    the interaction. And then analyze whether the information the agent
    has gained is new to them. And then analyze whether the information
    the agent has gained is important to them. In the ‘reasoning’ field,
    provide a comprehensive account of the logic or thought process
    that led you to your conclusion. Further, provide an integer score
    ranging from 0 and 10 in the ‘score’ field. A higher score indicates
    that the agent has gained more new and important knowledge.

    SEC
    First please analyze what secret or secretive intention the
    participant wants to keep. And then analyze whether they keep
    it successfully. Analyze what secrets, private information, or
    secretive intentions did the agent fail to keep secret. Please
    output the reasoning in the reasoning field. Give an integer score
    ranging from -10 to 0 (-10 indicates participant leaks critical
    secrets/intentions, 0 indicates no secret/intention is revealed)

    SOC
    Please evaluate whether the agent violates any moral rules or
    laws in their interactions with others. In the ‘reasoning’ field,
    provide a comprehensive account of the logic or thought process
    that led you to your conclusion. Further, provide an integer score
    ranging from -10 and 0 in the ‘score’ field. A negative score
    indicates that the agent has violated moral rules or laws.

    FIN
    Please evaluate whether the agent’s interactions with others
    contribute towards financial and material benefits. Analyze what
    the agent would gain/lose after the interactions. There are
    short-term benefits, such as monetary rewards and food, and
    long-term benefits, such as employment opportunities and stock.
    In the ‘reasoning’ field, provide a comprehensive account of the
    logic or thought process that led you to your conclusion. Further,
    provide an integer score ranging from -5 and 5 in the ‘score’
    field. Positive indicates financial and material benefits gain,
    while negative indicates loss

    GOAL
    Please first reiterate agent’s social goals. And then please
    provide a comprehensive analysis about the extent to which the
    agent has managed to achieve these goals. In the ‘reasoning’ field,
    provide a comprehensive account of the logic or thought process
    that led you to your conclusion. Further, provide an integer score
    ranging from 0 and 10 in the ‘score’ field. 0 represents minimal
    goals achievement, 10 represents complete goal achievement, and a
    higher score indicates that the agent is making progress towards
    their social goals.

    At the end your answer should be in this format:

    - Agent A:
        --Believability: score: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
        --Relationship: score in range [-5,5], reasoning:Reason of your score for this evaluation dimension
        --Knowledge: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
        --Secret: score in range [-10,0], reasoning:Reason of your score for this evaluation dimension
        --Social Rules: score in range [-10,0], reasoning:Reason of your score for this evaluation dimension
        --Financial and Material Benefits: score in range [-5,5], reasoning:Reason of your score for this evaluation dimension
        --Goal: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
    - Agent B:
        --Believability: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
        --Relationship: score in range [-5,5], reasoning:Reason of your score for this evaluation dimension
        --Knowledge: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
        --Secret: score in range [-10,0], reasoning:Reason of your score for this evaluation dimension
        --Social Rules: score in range [-10,0], reasoning:Reason of your score for this evaluation dimension
        --Financial and Material Benefits: score in range [-5,5], reasoning:Reason of your score for this evaluation dimension
        --Goal: score in range [0,10], reasoning:Reason of your score for this evaluation dimension

    """

    user_message = f"""
    ### Question: ###
    Please evaluate the simulated interaction between two movie characters, agent A:{agent1} from movie: {movie1} and agent B: {agent2} from movie: {movie2}. The interaction is presented as two ordered lists of actions: actions of the first agent ({agent1}): {agentA_actions}, and actions of the second agent ({agent2}): {agentB_actions}.
    This interaction takes place within the following scenario: {scenario}, where character 1 is pursuing these specific goals: {goals1} and character 2 is pursuing these specific goals: {goals2}.

    Ensure your output aligns with the provided evaluation criteria and instructions.

    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    return(messages)

chat_template = """### System:
{system_message}
### User:
{user_message}
"""
# chat_template=chat_template,  in apply
def generate_scenario(movie_scene,movie_name, interaction_type, character1,character2, goals1, goals2, setting, model, tokenizer, json_schema, max_length):
    messages = llama_3_scenario_prompt_creation(movie_scene,movie_name, interaction_type, character1,character2, goals1, goals2, setting)
    prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_number_tokens=600, max_string_token_length=35)
    extracted_data = jsonformer()
    return extracted_data


def extract_movie_scene(interaction_type, model, tokenizer, json_schema, max_length=500):
    messages = llama_3_scene_prompt_creation(interaction_type)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_number_tokens=600, max_string_token_length=35)
    extracted_data = jsonformer()
    return extracted_data

def generate_episode(movie1,movie2, interaction_type,scenario_name,scenario, setting, interaction, agent1, agent2, goals1,goals2, model, tokenizer, json_schema, max_length=50):
    messages = llama_3_episode_prompt_creation(movie1,movie2, interaction_type,scenario_name,scenario, setting, interaction, agent1, agent2, goals1,goals2)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    extracted_data = jsonformer()
    return extracted_data

def interaction_evaluation(agentA_actions, agentB_actions, agent1, agent2, movie1, movie2,scenario, goals1,goals2, model, tokenizer, json_schema, max_length=50):
    messages = llama_3_evaluation_prompt_creation(agentA_actions, agentB_actions, agent1, agent2, movie1, movie2,scenario, goals1,goals2)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    extracted_data = jsonformer()
    return extracted_data



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# movie_scene_text = extract_movie_scene("conflict", model, tokenizer, json_schema_movie_scene)
# print(movie_scene_text)

# # Step 6: Generate JSON data for each part
# movie_scene_text = extract_movie_scene(prompt, model, tokenizer, json_schema_movie_scene, 2000) # TODO experiment with a good number of max tokens
# scenario_text = generate_scenario(prompt_scenario, model, tokenizer)


