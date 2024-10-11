from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import yaml
from prompts_json import extract_movie_scene, generate_scenario, generate_episode, interaction_evaluation

from evaluation import evaluate_scene_v2
from utils import parse_yaml
import json
import pandas as pd
from tqdm import tqdm
import torch
import os
from concurrent.futures import ThreadPoolExecutor

# Clear GPU memory and set environment variable
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
print("___loading_config_____")
print("___defining_model_____")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Configurations
MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
# MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
interaction_types = ["exchange", "competition", "cooperation", "conflict", "coercion"]
scenes_per_type = 5  # Generating 2 scenes per interaction type


# Define JSON schemas
json_schema_movie_scene = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "character1": {"type": "string"},
        "character2": {"type": "string"},
        "setting": {"type": "string"},
        "relationship": {
            "type": "string",
            "enum": ["family", "friend", "romantic", "acquaintance", "stranger"]
        },
        "scenario": {"type": "string"},
        "goal1": {"type": "string"},
        "goal2": {"type": "string"}

    },
    "required": ["name", "scene", "character1","character2", "setting","relationship", "scenario", "goal1", "goal2"]
}

json_schema_scenario = {
    "type": "object",
    "properties": {
        "scenario": {"type": "string"},
        "goals1": {"type": "string"},
        "goals2": {"type": "string"},
    },
    "required": ["scenario", "goals1","goals2"]
}

print("___defining_model_____")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

quant_config = BitsAndBytesConfig(
    load_in_4bit=False,                # Switch to 8-bit quantization
    bnb_4bit_quant_type="nf4",         # Keep this, but it only applies to 4-bit, so no longer relevant here
    bnb_4bit_use_double_quant=True,    # Enable double quantization
    bnb_4bit_compute_dtype=torch.float32  # Use higher precision for computations
)

# Define the main model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", quantization_config=quant_config)
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Function to process each interaction type
def process_interaction_type(interaction_type):
    scenes_scenarios = []
    
    print(f"___Processing interaction type: {interaction_type}___")
    for i in range(scenes_per_type):
        # Extract the movie scene
        movie_scene = extract_movie_scene(interaction_type, model, tokenizer, json_schema_movie_scene, 1000)
        print(movie_scene)
        
        # Extract the scenario
        '''scenario = generate_scenario(movie_scene["scene"], movie_scene["name"], interaction_type,
                                     movie_scene["setting"], movie_scene["relationship"], model, tokenizer, json_schema_scenario, 1000)
        print(scenario)'''

        
        
        # Combine movie scene and scenario into one dictionary
        scene_scenario_data = {
            "interaction_type": interaction_type,
            "movie_scene": movie_scene,  
           
        }

        scenes_scenarios.append(scene_scenario_data)
    
    return scenes_scenarios

# Use ThreadPoolExecutor to parallelize processing
with ThreadPoolExecutor(max_workers=len(interaction_types)) as executor:
    futures = [executor.submit(process_interaction_type, interaction_type) for interaction_type in interaction_types]
    
    all_scenes_scenarios = []
    for future in futures:
        all_scenes_scenarios.extend(future.result())

# Create a DataFrame from the list of dictionaries
df = pd.json_normalize(all_scenes_scenarios)

# Save the DataFrame to a CSV file
df.to_csv("./pipeline_llama_new_70B.csv", index=False)

print("Finished generating scenes and scenarios.")
