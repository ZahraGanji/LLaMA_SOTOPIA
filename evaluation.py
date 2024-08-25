import torch
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# source ~/NLP/venv/bin/activate
#  source ~/localdisk/pythonvenv/bin/activate
def llama_3_judge_prompt_creation(description):

    json_format = {
        "explanation": {"Why do you think given movie description is real or not"},
        "is_real": {"Set to true if the movie scene is real. Set to false if movie scene is not real"},
    }

    system_message = f"""
    #### Persona: ###
    "You are a movie expert. You know every movie scene by heart. Your job is to check whether the input movie scene description is real or not.\
    If it's real - value of 'is_real' will be set to true, if it's not, it will be set to 'false'. Answer best to your movie knowledge. You will also state the reason for your choice"

    ### Goal: ###
    If it's real - value of 'is_real' will be set to true, if it's not, it will be set to 'false'. Answer best to your movie knowledge. You will also state the reason for your choice
    """

    user_message = f"""
    ### Question: ###
    Movie scene description: {description}

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
def evaluate_scene_v2(description, model, tokenizer, json_schema):
    messages = llama_3_judge_prompt_creation(description)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt, max_number_tokens=500,
                        max_array_length=500,
                        max_string_token_length=500)
    generated_data = jsonformer()
    return generated_data

agent1 = AutoModelForCausalLM.from_pretrained(model_name)
agent2 = AutoModelForCausalLM.from_pretrained(model_name)
agent3 = AutoModelForCausalLM.from_pretrained(model_name)
valuated_scene = "Two thugs are counting their loot after a robbery on a Gotham City rooftop. Suddenly, they hear a noise and look around nervously. Thug 1 says, ‘What was that?’ Thug 2 replies, ‘Probably just a rat. Let’s get out of here.’ Before they can move, a dark figure swoops down and lands in front of them. It’s Batman. Thug 1 asks, ‘Who are you?’ Batman grabs Thug 1 by the collar and lifts him off the ground, saying, ‘I’m Batman.’ Batman knocks out both thugs effortlessly and ties them up for the police. He then disappears into the night, leaving only the sound of his cape flapping in the wind."
evaluated_scene = "King Jumbo bumbo goes out on a marika island and handles out blue coconuts. Then a conflict arises because on of the island residents did not receive enough coconuts. The negotiation lasts for 3 hours before the kind kills all the island residents"
json_schema = {
            "type": "object",
            "properties": {
                "is_real": {"type": "boolean"},
                "explanation": {"type": "string"}
                }
            }
print(evaluate_scene_v2(evaluated_scene, agent1, tokenizer, json_schema))
print(evaluate_scene_v2(evaluated_scene, agent2, tokenizer, json_schema))
print(evaluate_scene_v2(evaluated_scene, agent3, tokenizer, json_schema))