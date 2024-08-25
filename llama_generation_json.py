from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 3: Load the pre-trained model and tokenizer
model_name = "databricks/dolly-v2-12b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Define individual prompts for each part
prompt_movie_scene = """
### Instruction ###
Please provide a detailed description of a movie scene that demonstrates the "exchange" type of human social interaction.
Include the following details:
- A description of the scene
- The characters involved
- The setting
- The nature of the conflict or interaction
"""

prompt_scenario = """
### Instruction ###
Based on the identified movie scene, please generate a scenario of human interactions and provide a detailed description of it. 
When generating scenarios, substitute the names of movie characters with more general terms such as "agents" to ensure broader applicability.
Include the following details:

1. A description of the scene
2. **Goal**: Describe the goal of the human interaction scenario.
3. **Constraints**: List the constraints or conditions related to the scenario.
4. **Interaction**: Describe the detailed interaction between the agents in the scenario.
"""

# Step 5: Define JSON schema
json_schema_movie_scene = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "characters_involved": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "scene_nature": {"type": "string"},
    },
    "required": ["description", "characters_involved", "setting", "scene_nature"]
}

json_schema_scenario = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "goal": {"type": "string"},
        "constraints": {"type": "array", "items": {"type": "string"}},
        "interaction": {"type": "string"},
    },
    "required": ["description", "goal", "constraints", "interaction"]
}

def llama_3_scene_prompt_creation(description):

    json_format = {
        "description": {"A consise description of a movie scene"},
        "characters_involved": {"List of characters involved in the movie scene"},
        "setting": {"Time and place where a scene occurs, including the physical environment and cultural context. It helps establish the mood and tone, influencing the story and characters."},
        "scene_nature": {"The nature of the scene. Can be: exchange, competition, cooperation, conflict, and coercion." }
        }

    system_message = f"""
    #### Persona: ###
    "You are a movie expert. Especially, you have a great memory of movie scenes. For every 5 types of human interaction: exchange, competition, cooperation, conflict, and coercion",
    you will find a detailed movie scene that encapsulates that interaction type.

    ### Goal: ###
    You will extract a movie scene based on the user input. You must include:
    - A description of the scene
    - The characters involved
    - The setting of the movie scene
    - The nature of the scene: exchange, competition, cooperation, conflict, and coercion
    """

    # TODO make a good prompt here 
    user_message = f""" 
    ### Question: ###
    

    ### Format: ###
    You should choose one of the options.
    Use the following json format:
    {json_format}
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    
    return(messages)

def generate_text(prompt, model, tokenizer, max_length=1000):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(
        inputs["input_ids"],
        max_length=1500,  # Adjust as needed
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_movie_scene(prompt, model, tokenizer, json_schema, max_length):
    messages = llama_3_scene_prompt_creation(description)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_number_tokens=max_length,
                        max_array_length=max_length,
                        max_string_token_length=max_length)
    generated_data = jsonformer()
    return generated_data

# Step 6: Generate JSON data for each part
movie_scene_text = generate_movie_scene(prompt, model, tokenizer, json_schema_movie_scene, 2000) # TODO experiment with a good number of max tokens
scenario_text = generate_text(prompt_scenario, model, tokenizer)

# Step 7: Print the generated JSON
import json

# Assuming you would like to manually structure the outputs as JSON
generated_data = {
    "movie_scene": {
        "description": movie_scene_text.split('\n')[0],  # Example extraction, adjust based on actual output
        "characters_involved": movie_scene_text.split('\n')[1],  # Example extraction
        "setting": movie_scene_text.split('\n')[2],  # Example extraction
        "conflict_nature": movie_scene_text.split('\n')[3]  # Example extraction
    },
    "scenario": {
        "description": scenario_text.split('\n')[0],  # Example extraction, adjust based on actual output
        "goal": scenario_text.split('\n')[1],  # Example extraction
        "constraints": scenario_text.split('\n')[2].split(', '),  # Example extraction
        "interaction": scenario_text.split('\n')[3]  # Example extraction
    }
}

print(json.dumps(generated_data, indent=2))
