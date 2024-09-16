from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml 

from llama_generation_json import extract_movie_scene, generate_scenario
from evaluation import evaluate_scene_v2
from utils import parse_yaml
import json
import pandas as pd
from tqdm import tqdm 
import torch
torch.cuda.empty_cache()

print("___loading_config_____")
# load config

MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
Model_judge = ["meta-llama/Meta-Llama-3-8B-Instruct","databricks/dolly-v2-12b","meta-llama/Meta-Llama-3.1-8B"]


# interaction_types = ["exchange", "competition", "cooperation", "conflict", "coercion"]
# scenes_per_type = 10
scenes_per_type = 1
interaction_types = ["exchange"]
judges_no = 3

# List to store all scenes and scenarios
all_scenes_scenarios = []


# define json schemas #TODO move to a separate file
json_schema_movie_scene = {
"type": "object",
    "properties": {
        "title": {"type": "string"},
        "description": {"type": "string"},
        "characters_involved": {"type": "array", "items": {"type": "string"}},
        "setting": {"type": "string"},
        "goals_and_motivations": {"type": "string"},
        "scene_nature": {"type": "string"},
    },
    "required": ["title", "description", "characters_involved", "setting","goals_and_motivations", "scene_nature"]
}

json_schema_scenario = {
    "type": "object",
    "properties": {
        "title":{"type": "string"},
        "description": {"type": "string"},
        "goal": {"type": "string"},
        "interaction": {"type": "string"},
    },
    "required": ["title","description", "goal", "interaction"]
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
# define the model
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load all judge models and their corresponding tokenizers
judge_models = []
judge_tokenizers = []
for judge_model_name in Model_judge:
    judge_model = AutoModelForCausalLM.from_pretrained(judge_model_name)
    judge_model.config.use_cache = False
    judge_model.config.pretraining_tp = 1
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    judge_models.append(judge_model)
    judge_tokenizers.append(judge_tokenizer)

# Loop over each interaction type
for interaction_type in interaction_types:
    print(f"___Processing interaction type: {interaction_type}___")
    for i in tqdm(range(scenes_per_type), desc=f"Generating scenes for {interaction_type}"):
        # Extract the movie scene
        movie_scene = extract_movie_scene(interaction_type, model, tokenizer, json_schema_movie_scene, 500)
        
        # Extract the scenario
        scenario = generate_scenario(movie_scene["description"], interaction_type, model, tokenizer, json_schema_scenario, 1000)
        
        # Validation logic: Use 3 different LLMs to validate the scene
        yes_count = 0
        for j in range(judges_no):
            # Load a model for each judge (you can reuse the same model or use different ones)
            judge_model = judge_models[j]
            judge_tokenizer = judge_tokenizers[j]
            answer = evaluate_scene_v2(movie_scene["description"], judge_model, judge_tokenizer, json_schema_evaluation)
            if answer["is_real"]:
                yes_count += 1
        
        # If at least 2 out of 3 judges say "yes", the scene is marked as valid
        validation_result = "yes" if yes_count >= 2 else "no"
        
        # Combine movie scene and scenario into one dictionary
        scene_scenario_data = {
            "interaction_type": interaction_type,
            "movie_scene": movie_scene,
            "scenario": scenario,
            "validation": validation_result
        }
        
        # Append the data to the list
        all_scenes_scenarios.append(scene_scenario_data)

# Create a DataFrame from the list of dictionaries
df = pd.json_normalize(all_scenes_scenarios)

# Save the DataFrame to a CSV file
df.to_csv("scenes_and_scenarios.csv", index=False)








