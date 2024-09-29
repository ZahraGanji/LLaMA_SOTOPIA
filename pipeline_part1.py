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

# Configurations
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
# MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
Model_judge = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "databricks/dolly-v2-12b", "databricks/dolly-v2-3b"]
interaction_types = ["exchange", "competition", "cooperation", "conflict", "coercion"]
scenes_per_type = 10  # Generating 2 scenes per interaction type
judges_no = 3

# Define JSON schemas
json_schema_movie_scene = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "character1": {"type": "string"},
        "character2": {"type": "string"},
        "setting": {"type": "string"},
        "goals1": {"type": "string"},
        "goals2": {"type": "string"}
    },
    "required": ["title", "description", "character1","character2", "setting", "goals1", "goals2"]
}

json_schema_scenario = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "goals1": {"type": "string"},
        "goals2": {"type": "string"},
        "interaction": {"type": "string"},
    },
    "required": ["title", "description", "goal1","goal2", "interaction"]
}


json_schema_evaluation = {
    "type": "object",
    "properties": {
        "is_real": {"type": "boolean"},
        "explanation": {"type": "string"}
    }
}


print("___defining_model_____")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# If you want to load the model explicity
quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    )

# Define the main model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", quantization_config=quant_config)

model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load all judge models and their corresponding tokenizers
judge_models = []
judge_tokenizers = []
for judge_model_name in Model_judge:
    judge_model = AutoModelForCausalLM.from_pretrained(judge_model_name, device_map="auto", torch_dtype=torch.float16)
    judge_model.config.use_cache = False
    judge_model.config.pretraining_tp = 1
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    judge_models.append(judge_model)
    judge_tokenizers.append(judge_tokenizer)


# Function to process each interaction type
def process_interaction_type(interaction_type):
    scenes_scenarios = []
    
    print(f"___Processing interaction type: {interaction_type}___")
    for i in range(scenes_per_type):
        # Extract the movie scene
        movie_scene = extract_movie_scene(interaction_type, model, tokenizer, json_schema_movie_scene, 1000)
        print(movie_scene)
        
        # Extract the scenario
        scenario = generate_scenario(movie_scene["description"], movie_scene["title"], interaction_type,
                                     movie_scene["character1"], movie_scene["character2"],movie_scene["goals1"],movie_scene["goals2"],
                                     movie_scene["setting"], model, tokenizer, json_schema_scenario, 1000)
        print(scenario)

        
        
        # Validation logic: Use 3 different LLMs to validate the scene
        yes_count = 0
        for j in range(judges_no):
            # Load a model for each judge (you can reuse the same model or use different ones)
            answer = evaluate_scene_v2(movie_scene["description"],movie_scene["title"], judge_models[j], judge_tokenizers[j], json_schema_evaluation)
            if answer["is_real"]:
                yes_count += 1
        
        # If at least 2 out of 3 judges say "yes", the scene is marked as valid
        validation_result = "yes" if yes_count >= 2 else "no"

       
        
        # Combine movie scene and scenario into one dictionary
        scene_scenario_data = {
            "interaction_type": interaction_type,
            "movie_scene": movie_scene,
            "scenario": scenario,
            "validation": validation_result,       
           
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
df.to_csv("./pipeline_total.csv", index=False)

print("Finished generating scenes and scenarios.")
