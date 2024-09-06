from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml 

from llama_generation_json import extract_movie_scene, generate_scenario
from evaluation import evaluate_scene_v2
from utils import parse_yaml
import json
import torch


print("___loading_config_____")
# load config
# config = parse_yaml("config.yaml")
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
prompt = "Two thugs are counting their loot after a robbery on a Gotham City rooftop. Suddenly, they hear a noise and look around nervously. Thug 1 says, ‘What was that?’ Thug 2 replies, ‘Probably just a rat. Let’s get out of here.’ Before they can move, a dark figure swoops down and lands in front of them. It’s Batman. Thug 1 asks, ‘Who are you?’ Batman grabs Thug 1 by the collar and lifts him off the ground, saying, ‘I’m Batman.’ Batman knocks out both thugs effortlessly and ties them up for the police. He then disappears into the night, leaving only the sound of his cape flapping in the wind."
# interaction_type = config["interaction_type"]
interaction_type = "exchange"
judges_no = 3

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
print("___defining_model_____")
# define the model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("____extracting_movie_scene_____")
# extract the movie scene
movie_scene = extract_movie_scene(interaction_type, model, tokenizer, json_schema_movie_scene, 500)
# generate the scenario
# scenario = generate_scenario(movie_scene["description"], interaction_type, model, tokenizer, json_schema_scenario, 2000)
# print(scenario)
# evaluate whether the scenario is real or made up
# scenario_judges = [AutoModelForCausalLM.from_pretrained(MODEL_NAME) for _ in range(judges_no)]
# json_schema_evaluation = { #TODO move into a separate file
#         "type": "object",
#         "properties": {
#             "is_real": {"type": "boolean"},
#             "explanation": {"type": "string"}
#             }
#     }
# yes_count = 0
# for i in range(judges_no):
#     anwser = evaluate_scene_v2(movie_scene["description"], scenario_judges[i], tokenizer, json_schema_evaluation)
#     if anwser["is_real"]:
#         yes_count+=1

# if yes_count >= 2:
#     print("Generated movie scenario exists according to Llama-3. Proceeding to the next step")
# else:
#     print("Generated movie scenario does not exist according to Llama-3")






