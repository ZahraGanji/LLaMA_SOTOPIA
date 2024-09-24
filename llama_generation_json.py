from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
torch.cuda.empty_cache()

# Step 5: Define JSON schema
json_schema_movie_scene = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "characters_involved": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "goals_and_motivations": {"type": "string"},
        "scene_nature": {"type": "string"},
    },
    "required": ["description", "characters_involved", "setting","goals_and_motivations", "scene_nature"]
}

json_schema_scenario = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "goal": {"type": "string"},
        "interaction": {"type": "string"},
    },
    "required": ["description", "goal", "interaction"]
}


def llama_3_scene_prompt_creation(interaction_type):
    print("in llama_3_scene_prompt_creation")
    json_format = {
        "title": "Title of the movie",
        "description": "A description of the movie scene",
        "characters_involved": ["List of all characters involved in the movie scene."],
        "setting": "Time and place where the scene occurs, including the physical environment and cultural context. This helps establish the mood and tone, influencing the story and characters.",
        "goals_and_motivations": "The goals and motivations of each character in the scene"
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
    - Title of the movie
    - A vivid description of the scene which describes the type of human social interaction correctly.
    - List of all characters involved in the movie scene.
    - The setting of the movie scene. Time and place where the scene occurs, including the physical environment and cultural context. This helps establish the mood and tone, influencing the story and characters.
    - The goals and motivations of each character involved in the scene. 
    """

    # TODO make a good prompt here 
    user_message = f""" 
    ### Question: ###
    Please find a movie scene that demonstrates the {interaction_type} type of human social interaction. 
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

def llama_3_scenario_prompt_creation(movie_scene,movie_name, interaction_type, characters, goals, setting):
    print("in llama_3_scenario_prompt_creation")
    
    json_format = {
        "title": "Title of the scenario.",
        "description": "A vivid description of the scenario.",
        "setting": "Describtion of the time, place, and context of the scenario.", 
        "goal": "The social goals of each agent involved in the scenario base on the goals of the characters of the given movie scene. When creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.",
        "interaction": "Provide a detailed description of how the agents interact with one another in the scenario."
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
    The scenario should depict a meaningful social interaction between two agents (Agent 1 and Agent 2), highlighting their relationship—whether they are strangers, acquaintances, friends, romantic partners, or family members—and the purpose behind each agent’s interaction with the other.. 
    Please avoid mentioning specific names in the scenario and keep all the mentions gender-neutral.
    Include the following details in your scenario description:
    - Title: give a title to the generated scenario.
    - description: A vivid description of the scenario.
    - Setting: Describe the time, place, and context of the scenario.
    - Goal: The social goals of each agent involved in the scenario base on the goals of the characters of the given movie scene.
    When creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively resolve it.
    - Interaction: Provide a detailed description of how the agents interact with one another in the scenario.
    """
    
    user_message = f""" 
    ### Question: ###
    Based on the movie scene {movie_scene} from {movie_name}, which showcases a {interaction_type} type of human interaction, featuring the characters {characters} with their respective goals and motivations {goals}, and taking place in the setting {setting}, please generate a detailed description of the human interaction scenario.
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

def llama_3_episode_prompt_creation(movie_name, interaction_type,scenario_name,scenario, setting, interaction, agent1, agent2, goals):
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
    When a user presents a scenario with characters, their social objectives, and the setting, you simulate interactions based on their profiles from specific movie scenes, incorporating their traits such as age, gender, moral values, and secrets. As a film expert, you draw upon your knowledge of these characters to create authentic interactions aligned with their personalities and objectives.

    In this simulation, characters are referred to as agents, and they take turns interacting in a round-robin fashion (e.g., Agent 1 acts first, followed by Agent 2, and so on). On their turn, each agent can choose one of the following actions:

    Speak (verbal communication),
    Use non-verbal communication (e.g., smiling, hugging),
    Take a physical action (e.g., playing music).
    These are key components of social interaction, but agents may also opt to do nothing (remaining silent) or decide to leave, thereby ending the episode.

    Each interaction is limited to a maximum of 20 turns, as most tasks can typically be resolved within this range. The episode concludes when one agent chooses to leave or the turn limit is reached.
   
    Create an episode with this format:
    - Agent A: List of the first agents actions with respect to the order of interactions.
    - Agent B: List of the second agents actions with respect to the order of interactions.

    """
 
    user_message = f""" 
    ### Question: ###
    Based on the scenario {scenario_name}, set in the context of {scenario} and the location {setting}, featuring the interaction {interaction} between agents, which exemplifies the {interaction_type} type of human interaction, we have the following characters: Agent 1: {agent1} and Agent 2: {agent2}, from the movie {movie_name}. Their objectives are: {goals}.
    Please simulate an interaction between the agents.
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

def generate_scenario(movie_scene,movie_name, interaction_type, characters, goals, setting, model, tokenizer, json_schema, max_length):
    messages = llama_3_scenario_prompt_creation(movie_scene,movie_name, interaction_type, characters, goals, setting)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_number_tokens=max_length,
                        max_array_length=max_length,
                        max_string_token_length=max_length)
    extracted_data = jsonformer()
    return extracted_data


def extract_movie_scene(interaction_type, model, tokenizer, json_schema, max_length=500):
    messages = llama_3_scene_prompt_creation(interaction_type)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt,max_number_tokens=max_length,
                        max_array_length=max_length,
                        max_string_token_length=max_length)
    extracted_data = jsonformer()
    return extracted_data

def generate_episode(movie_name, interaction_type,scenario_name,scenario, setting, interaction, agent1, agent2, goals, model, tokenizer, json_schema, max_length=500):
    messages = llama_3_episode_prompt_creation(movie_name, interaction_type,scenario_name,scenario, setting, interaction, agent1, agent2, goals)
    # print(messages)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt,max_number_tokens=max_length,
                        max_array_length=max_length,
                        max_string_token_length=max_length)
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


