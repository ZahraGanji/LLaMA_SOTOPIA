from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonformer import Jsonformer
import yaml 

from llama_generation import generate_response
from evaluation import evaluate_scene_v2
from utils import parse_yaml
import json
import torch


if __name__ == "__main__":
    # load config
    config = parse_yaml("config.yaml")
    MODEL_NAME = config["MODEL_NAME"]
    prompt = config["prompt"]
    judges_no = config["judges_no"]

    # define the model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # generate the movie scenario
    movie_scenario = generate_movie_scene(prompt)

    # evaluate whether the scenario is real or made up
    scenario_judges = [AutoModelForCausalLM.from_pretrained(MODEL_NAME) for _ in range(config["judges_no"])]
    json_schema_evaluation = {
            "type": "object",
            "properties": {
                "is_real": {"type": "boolean"},
                "explanation": {"type": "string"}
                }
        }
    yes_count = 0
    for i in range(judges_no):
        anwser = evaluate_scene_v2(movie_scenario, scenario_judges[i], tokenizer)
        if anwser["is_real"]:
            yes_count+=1
    
    if yes_count >= 2:
        print("Generated movie scenario exists according to Llama-3. Proceeding to the next step")
    else:
        print("Generated movie scenario does not exist according to Llama-3")



    


