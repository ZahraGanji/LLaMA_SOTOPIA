from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml 

from llama_generation_json import extract_movie_scene, generate_scenario
from evaluation import evaluate_scene_v2
from utils import parse_yaml
import json
import pandas as pd
from tqdm import tqdm 
import torch


print("___loading_config_____")
# load config
<<<<<<< HEAD

MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B"
Model_judge = ["meta-llama/Meta-Llama-3-8B-Instruct","databricks/dolly-v2-12b","meta-llama/Meta-Llama-3.1-8B"]


# interaction_types = ["exchange", "competition", "cooperation", "conflict", "coercion"]
# scenes_per_type = 10
scenes_per_type = 1
interaction_types = ["exchange"]
=======
# config = parse_yaml("config.yaml")
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
JUDGE_NAME_1 = "meta-llama/Meta-Llama-3.1-8B-Instruct"
JUDGE_NAME_2 = "meta-llama/Meta-Llama-3.1-70B-Instruct"
JUDGE_NAME_3 = "bigscience/bloom"
prompt = "Two thugs are counting their loot after a robbery on a Gotham City rooftop. Suddenly, they hear a noise and look around nervously. Thug 1 says, ‘What was that?’ Thug 2 replies, ‘Probably just a rat. Let’s get out of here.’ Before they can move, a dark figure swoops down and lands in front of them. It’s Batman. Thug 1 asks, ‘Who are you?’ Batman grabs Thug 1 by the collar and lifts him off the ground, saying, ‘I’m Batman.’ Batman knocks out both thugs effortlessly and ties them up for the police. He then disappears into the night, leaving only the sound of his cape flapping in the wind."
# interaction_type = config["interaction_type"]
interaction_type = "exchange"
>>>>>>> e3bfb19e4266d9484a5076c86a02692d461d08d5
judges_no = 3

# List to store all scenes and scenarios
all_scenes_scenarios = []


# define json schemas #TODO move to a separate file
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

<<<<<<< HEAD
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


=======
evaluation_json_schema = {
            "type": "object",
            "properties": {
                "is_real": {"type": "boolean"},
                "explanation": {"type": "string"}
                }
            }

print("___defining_model_____")
# define the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("____extracting_movie_scene_____")
# extract the movie scene
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
movie_scene = extract_movie_scene(interaction_type, model, tokenizer, json_schema_movie_scene, 1000)
# generate the scenario
scenario = generate_scenario(movie_scene["description"], interaction_type, model, tokenizer, json_schema_scenario, 1000)
print(scenario)

tk_list=[]
judge_list = []
judge_1_tokenizer = AutoTokenizer.from_pretrained(JUDGE_NAME_1)
judge_model_1 = AutoModelForCausalLM.from_pretrained(JUDGE_NAME_1)
judge_model_1 = judge_model.to(device)

judge_2_tokenizer = AutoTokenizer.from_pretrained(JUDGE_NAME_2)
judge_model_2 = AutoModelForCausalLM.from_pretrained(JUDGE_NAME_2)
judge_model_2 = judge_model.to(device)

judge_3_tokenizer = AutoTokenizer.from_pretrained(JUDGE_NAME_3)
judge_model_3 = AutoModelForCausalLM.from_pretrained(JUDGE_NAME_3)
judge_model_3 = judge_model.to(device)


judge_list = [judge_model_1, judge_model_2, judge_model_3]
tk_list = [judge_1_tokenizer, judge_2_tokenizer, judge_3_tokenizer]
real_count=0

for i in range(3):
    data = evaluate_scene_v2(movie_scene["description"], judge_list[i], tk_list[i], evaluation_json_schema)
    if data["is_real"]:
        real_count+=1

if real_count>=2:
    print("Selected movie scene is real")
else:
    print("Selected movie scene is not real")
>>>>>>> e3bfb19e4266d9484a5076c86a02692d461d08d5






