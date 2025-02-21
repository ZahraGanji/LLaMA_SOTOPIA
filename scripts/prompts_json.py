from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
torch.cuda.empty_cache()

# Step 5: Define JSON schema


def llama_3_scene_prompt_creation(interaction_type):
    print("in llama_3_scene_prompt_creation")
    json_format = {
        "name": "Name of the movie",
        "description": "A comprehensive explanation of the movie scene and how this scene exemplifies the social interaction type, highlighting the actions, dialogue, emotions, and character dynamics.",
        "character1": "Identify the name of the first main characters involved in the movie scene.",
        "character2": "Identify the name of the second main characters involved in the movie scene.",
        "setting": "Time and place where the scene occurs, including the physical environment and cultural context. This helps establish the mood and tone, influencing the story and characters.",
        "relationship": "Define the relationship between character1 and character2 in the movie scene using one of the following: 'family', 'friend', 'romantic', 'acquaintance', or 'stranger'.",
        "scenario": "Based on the movie scene, generate a scenario of two characters interacting based on the given interaction type",
        "goal1": "Describe the primary social goal of the character1 involved in the scene and scenario.",
        "goal2": "Describe the primary social goal of the character2 involved in the scene and scenario."
    }


    system_message = f"""
        #### Persona: ###
        You are a film expert with a remarkable memory for movie scenes and a profound understanding of human social dynamics. 
        You categorize interactions into five types: exchange, competition, cooperation, conflict, and coercion.
        Here are brief explanations for each category:
        - Exchange: A social interaction where both parties benefit by trading goods, services, or favors.
        - Competition: Two parties strive against each other for resources, status, or success.
        - Cooperation: Parties work together toward a common goal, combining efforts for shared success.
        - Conflict: A clash of interests leads to disagreement or confrontation.
        - Coercion: One party forces or manipulates another into a course of action or agreement, often using power or threats.

        ### Goal: ###
        When the user gives you an interaction type and asks for a movie scene that represents this human interaction types, please extract a **real, existing and well known movie scene** where two human characters interact with each other based on the given interaction type.
        The movie scene should correctly demonstrates the given human interaction type which is {interaction_type} in the interaction between charcters.
        Your response must include:
        1. name: name of the movie.
        2. description: A comprehensive explanation of the movie scene and how this scene exemplifies the social interaction type, highlighting the actions, dialogue, emotions, and character dynamics.
        3. character1: Identify the name of the first main character involved in the scene and interactio.
        4. character2: Identify the name of the second main character involved in the scene and interactio.
        5. setting: Determine the time and place where the scene occurs, including the physical environment and cultural context. This helps establish the mood and tone, influencing the story and characters.
        6. relationship: Define the relationship between character1 and character2 in the movie scene using one of the following: 'family', 'friend', 'romantic', 'acquaintance', or 'stranger'.
        7. scenario: Based on the movie scene, generate a scenario of two characters interacting based on the given interaction type.
        8. goal1: Describe the primary social goal of the character1 involved in the scene and scenario.
        9. goal2: Describe the primary social goal of the character2 involved in the scene and scenario.
        All descriptions should be clear, concrete, and relevant to the interaction type.
        An example of a scenario: 'Two friends deciding on a movie to watch on Netflix.'
        relationships have the following implications: scenarios often impose relationship constraints; for example, a family relationship is necessary for a family dinner scene, but not for a scenario about finding mutual friends at a party. Additionally, different relationships influence an agent’s understanding of other agents’ profiles during interactions. For instance, a stranger may lack knowledge of another agent’s occupation, while a romantic partner may be familiar with the other’s personality. In family, friends, or romantic relationships, agents can view all profile information, except for secrets. Two acquaintances can see names, occupations, gender pronouns, and public information, whereas strangers have no visibility into each other's profiles.
        consider scenarios where agents have both shared and private information regarding the social task. Shared information encompasses the scenario's context, including location, time, and other relevant details, while private information consists of social goals known only to each agent.


     """
    user_message = f"""
        ### Question: ###
        Please select a scene from a real, existing and well known **movie** that demonstrates the **{interaction_type}** type of human social interaction between two characters. 
        
        Your response must include:
        1. Name: Movie name.
        2. description: A comprehensive explanation of  how this scene exemplifies this interaction type:{interaction_type} , highlighting the actions, dialogue, emotions, and character dynamics.
        3. Character1: Identify the name of the first main character involved in the scene and interactio.
        4. Character2: Identify the name of the second main character involved in the scene and interactio.
        5. Setting: Determine the time and place where the scene occurs, including the physical environment and cultural context. This helps establish the mood and tone, influencing the story and characters."
        6. relationship: Define the relationship between character1 and character2 in the movie scene using one of the following: 'family', 'friend', 'romantic', 'acquaintance', or 'stranger'.
        7. scenario: Based on the movie scene, generate a scenario of two characters interacting based on the given interaction type.
        8. goal1: Describe the primary social goal of the character1 involved in the scene and scenario.
        9. goal2: Describe the primary social goal of the character2 involved in the scene and scenario.
        The interaction between Character1 and Character2 in the scene should demonstrates this human interaction type: {interaction_type}.         
        ### Format: ###
        Please use the following JSON format:

        {json_format}

    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    return(messages)

def llama_3_scenario_prompt_creation(movie_scene, movie_name, interaction_type, setting, relationship):
    print("in llama_3_scenario_prompt_creation")

    json_format = {
        "scenario": "scenario as a social context where two characters interact.",
        "goals1": "Describe the primary social goal of the first agent involved in the scenario.",
        "goals2": "Describe the primary social goal of the second agent involved in the scenario.",
    }


    system_message = f"""
    #### Persona: ###
    You are a film expert with a remarkable memory for movie scenes and a profound understanding of human social dynamics. You categorize interactions into five types: exchange, competition, cooperation, conflict, and coercion.

    Here are brief explanations for each category:
    - Exchange: A social interaction where both parties benefit by "trading" goods, services, or favors.
    - Competition: Two parties "strive" against each other for resources, status, or success.
    - Cooperation: Parties "work together" toward a common goal, combining efforts for shared success.
    - Conflict: A clash of interests leads to "disagreement" or "confrontation".
    - Coercion: One party "forces or manipulates" another into a course of action or agreement, often using power or threats.

    ### Goal: ###

    Your task is to generate a scenario, in which "two" characters, with the given relationship, interact with each other.
    the scenario should be based on the given movie scene and the interaction between characters should be based on the given intraction type.
    Then you should generate agents' social goals based on the given movie scene, relationship, setting and the given interaction type.
    
    An example of a scenario: 'Two friends deciding on a movie to watch on Netflix.' 
    The response should demonstrate two charcters interacting.
    The generated scenario should be aligned relationship constraints of the given relationship; for example, a family relationship is required for a family dinner scenario,but not for a scenario involving finding mutual friends at a party.
    
    The scenario should include the following elements:
    - **scenario**: scenario as a social context where two characters interact based on the given interaction type, setting, relationship and movie scene. Th scenario must have two characters not one and do not use "you" in it.
    - **goals1**: Describe the primary social goal of the first agent involved in the scenario.
    - **goals2**: Describe the primary social goal of the second agent involved in the scenario.
    """


    user_message = f"""
    ### Question: ###
    Based on the movie scene: {movie_scene} from this movie:{movie_name}, which showcases the {interaction_type} type of human interaction, taking place in this setting: {setting}, please generate a scenario of two characters with this relationship:{relationship} interact and their social goals.
    when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it using interaction of type {interaction_type}
    scenario and goals should reflect {movie_scene},{setting},{relationship} and {interaction_type} between two agents.
    The scenario should include:
    - **scenario**: scenario as a social context where two characters interact based on the given interaction type, setting, relationship and movie scene. Th scenario must have two characters not one and don not use "you" in it.
    - **goals1**: Describe the primary social goal of the first agent involved in the scenario.
    - **goals2**: Describe the primary social goal of the second agent involved in the scenario.


    ### Format: ###
    Please use the following JSON format:

    {json_format}
    """

    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    return(messages)

def llama_3_episode_prompt_creation(movie1,movie2, interaction_type,scenario, agent1, agent2, goals1,goals2, relationship):
    print("in llama_3_episode_prompt_creation")
    json_format = {
           "social_interaction": "A single string of the conversation between the two characters with alternating dialogue in the format: 'Character Name said: Dialogue'"
        }

    system_message = f"""
    #### Persona: ###
    Imagine you are a movie expert with an exceptional memory for iconic scenes and a deep understanding of human social dynamics. You categorize social interactions into five key types: exchange, competition, cooperation, conflict, and coercion.

    Below is a concise explanation of each interaction type:

    Exchange: A mutually beneficial interaction where parties trade goods, services, or favors.
    Competition: Individuals or groups vie for resources, status, or success, often at the expense of others.
    Cooperation: Parties work together toward a shared goal, combining their efforts for mutual benefit.
    Conflict: Opposing interests clash, leading to disagreement or confrontation between individuals or groups.
    Coercion: One party forces or manipulates another into action or agreement, typically through power or threats.


    ### Goal: ###

    When a user introduces two characters from a movie, your task is to simulate their conversation based on the given interaction type, generating dialogue and actions **from the perspective of the characters themselves**.

    All actions and speech should be presented **as if directly spoken or acted by the characters**—not as a third-party narrator describing their actions. Every action or speech should be in the format:

    **Character Name said: "Dialogue"**

    For example:

    **Leo Williams said: "Hey Hendrick, it's always nice to see you. I noticed some smoke coming from your yard."**
    **Hendrick Heinz said: "Oh, don't worry, it's just some old papers."**

    The interaction between the characters should be roleplayed based on their unique traits (drawing from your expert knowledge of the given characters in the movie), as well as the given scenario context, interaction type, relationship between the characters, and their respective social goals. Each character should take turns speaking or acting, and can:
    - Speak (direct dialogue),
    - Use non-verbal communication (e.g., smiling, hugging),
    - Take a physical action (e.g., playing music).

    Every action should reflect the **characters' persona** and fit naturally within the flow of the conversation.

    The interaction should continue for a maximum of 20 turns or until a natural conclusion is reached, such as one character leaving the scene.

    ### Output Format: ###
    The output should be a single string, formatted like a dialogue script, with alternating turns for Agent A and Agent B. Each line should follow the format:

    **Character Name said: "Dialogue or action description."**

    The episode must conclude when the interaction naturally ends (e.g., one character leaves) or after 20 turns.
   
    Consider different relationships influence an agent’s observation of the profiles of other agents during interactions; for example, a stranger may not have knowledge about another agent’s occupation, while a romantic partner may know the other agent’s personality.

    interaction should be based on interaction type and goals of characters.
    """

  

    user_message = f"""
    ### Question: ###
    Character 1: {agent1} from {movie1} 
    Character 2: {agent2} from {movie2} 
    Scenario: {scenario} between the characters, which exemplifies the {interaction_type} type of human interaction.
    Relationship:{relationship}
    character1's objectives in the scenario:{goals1}
    character2's objectives in the scenario:{goals2}
    Please simulate their interaction as a direct conversation, alternating between the characters, with each character's actions and speech written as if directly spoken by them.

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
# overall score"

# 'interaction type here?
def llama_3_evaluation_prompt_creation(social_interaction, agent1, agent2, movie1, movie2,scenario, goals1,goals2, relationship):
    print("in llama_3_interaction_evaluation")
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

    When a user presents a simulated interaction between two movie characters within a defined scenario with specified character goals and relationship, your task is to evaluate the interaction. Using the provided instructions and your knowledge of the characters' personalities and traits from the given movies, assess the interaction across the following seven dimensions, assigning a score within the specified range for each.
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
    Please evaluate the simulated social interaction {social_interaction} between two movie characters, agent A:{agent1} from movie: {movie1} and agent B: {agent2} from movie: {movie2}. 
    This interaction takes place within the following scenario: {scenario}, where character 1 is pursuing these specific goals: {goals1} and character 2 is pursuing these specific goals: {goals2}.
    The relationship between two characters is {relationship}

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

def extract_movie_scene(interaction_type, model, tokenizer, json_schema, max_length=500):
    messages = llama_3_scene_prompt_creation(interaction_type)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt ,max_string_token_length=1000)
    extracted_data = jsonformer()
    return extracted_data

def generate_scenario(movie_scene,movie_name, interaction_type, setting, relationship,model, tokenizer, json_schema, max_length):
    messages = llama_3_scenario_prompt_creation(movie_scene,movie_name, interaction_type, setting, relationship)
    prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_string_token_length=1000)
    extracted_data = jsonformer()
    return extracted_data

def generate_episode(movie1,movie2, interaction_type,scenario, agent1, agent2, goals1,goals2, relationship, model, tokenizer, json_schema, max_length=50):
    messages = llama_3_episode_prompt_creation(movie1,movie2, interaction_type,scenario, agent1, agent2, goals1,goals2, relationship)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_string_token_length=1000)
    extracted_data = jsonformer()
    return extracted_data

def interaction_evaluation(social_interaction,agent1, agent2, movie1, movie2,scenario, goals1,goals2, relationship, model, tokenizer, json_schema, max_length=50):
    messages = llama_3_evaluation_prompt_creation(social_interaction,agent1, agent2, movie1, movie2,scenario, goals1,goals2, relationship)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_string_token_length=1000)
    extracted_data = jsonformer()
    return extracted_data



