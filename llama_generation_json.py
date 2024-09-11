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
        "characters_involved": ["List of all characters involved in the movie scene"],
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
    - The key characters involved in the movie scene.
    - The setting of the movie scene.
    - The goals and motivations of each character involved in the scene. 
    """

    # TODO make a good prompt here 
    user_message = f""" 
    ### Question: ###
    Please find a movie scene that demonstrates the {interaction_type} type of human social interaction. Your output should be a description of a movie
    scene with accordance to your instructions.

    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    
    return(messages)

def llama_3_scenario_prompt_creation(movie_scene, interaction_type):
    print("in llama_3_scenario_prompt_creation")
    json_format = {
        "title": "Title of the scenario.",
        "description": "A description of the scenario.",
        "setting": "Describe the time, place, and context of the scenario.", 
        "goal": "Explain the objectives of each agent involved.",
        "interaction": "Provide a detailed account of how the agents interact with one another."
        }
        


    system_message = f"""
    #### Persona: ###
    Imagine you are a movie expert with an exceptional memory for film scenes and a deep
    understanding of human social interactions. You categorize these interactions into five key types:
    exchange, competition, cooperation, conflict, and coercion.
    Here is a consice explanation about each of the categories for your guidance: 
    Exchange: A social interaction where parties trade goods, services, or favors, with mutual benefit.
    Competition: Individuals or groups vie against each other for resources, status, or success.
    Cooperation: Parties work together toward a common goal, sharing effort and benefits.
    Conflict: Opposing interests clash, leading to disagreement or confrontation between individuals or groups.
    Coercion: One party forces or manipulates another into action or agreement, often using power or threats.

    ### Goal: ###
    When the user requests you to generate a scenario based on the movie scene and human interaction type he gives you, generate a human interaction scenario based on the given movie scene and given interaction type
    and use general terms like "agents" instead of specific character names to ensure broader applicability.
    Include the following details in your scenario description:
    - Title: give a title to the generated scenario.
    - description: A vivid description of the scenario.
    - Setting: Describe the time, place, and context of the scenario.
    - Goal: Explain the objectives of the agents involved in the scenario.
    - Interaction: Provide a detailed description of how the agents interact with one another in the scenario.
    """

    
    user_message = f""" 
    ### Question: ###
    Based on the movie scene {movie_scene}, which exemplifies the {interaction_type} type of human interaction, please generate a detailed description of a human interaction scenario.
    Your output should be a description of a scenario with accordance to your instructions.
    ### Format: ###
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    
    return(messages)

def generate_scenario(movie_scene_description, interaction_type, model, tokenizer, json_schema, max_length):
    messages = llama_3_scenario_prompt_creation(movie_scene_description, interaction_type)
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


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# movie_scene_text = extract_movie_scene("conflict", model, tokenizer, json_schema_movie_scene)
# print(movie_scene_text)

# # Step 6: Generate JSON data for each part
# movie_scene_text = extract_movie_scene(prompt, model, tokenizer, json_schema_movie_scene, 2000) # TODO experiment with a good number of max tokens
# scenario_text = generate_scenario(prompt_scenario, model, tokenizer)


