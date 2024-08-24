import torch
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# source ~/NLP/venv/bin/activate
#  source ~/localdisk/pythonvenv/bin/activate
def evaluate_scene_v2(description, model, tokenizer):
    # messages = [
    # {"role": "system", "content": """You are a movie expert. You know every movie scene by heart. Your job is to check whether the input movie scene description is real or not.\
    #  If it's real - value of 'is_real' will be set to true, if it's not, it will be set to 'false'. Answer best to your movie knowledge."""},
    # {"role": "user", "content": f"{description}"},
    # ]

    messages = f"You are a movie expert. You know every movie scene by heart. Your job is to check whether the input movie scene description is real or not.\
    #  If it's real - value of 'is_real' will be set to true, if it's not, it will be set to 'false'. Answer best to your movie knowledge. You will also state the reason for your choice. Movie scene description {description}"

    json_schema = {
    "type": "object",
    "properties": {
        "is_real": {"type": "boolean"},
        "explanation": {"type": "string"},
    }}

    # prompt = "Generate a person's information based on the following schema:"
    jsonformer = Jsonformer(model, tokenizer, json_schema, messages)
    generated_data = jsonformer()
    return generated_data
# pipeline1 = pipeline(
#     "text-generation",
#     model=model_name,
#     model_kwargs={"torch_dtype": torch.bfloat16})
# pipeline2 = pipeline(
#     "text-generation",
#     model=model_name,
#     model_kwargs={"torch_dtype": torch.bfloat16})
# pipeline3 = pipeline(
#     "text-generation",
#     model=model_name,
#     model_kwargs={"torch_dtype": torch.bfloat16})
agent1 = AutoModelForCausalLM.from_pretrained(model_name)
agent2 = AutoModelForCausalLM.from_pretrained(model_name)
agent3 = AutoModelForCausalLM.from_pretrained(model_name)
# Create a pipeline for text classification

# evaluated_scene = "Two thugs are counting their loot after a robbery on a Gotham City rooftop. Suddenly, they hear a noise and look around nervously. Thug 1 says, ‘What was that?’ Thug 2 replies, ‘Probably just a rat. Let’s get out of here.’ Before they can move, a dark figure swoops down and lands in front of them. It’s Batman. Thug 1 asks, ‘Who are you?’ Batman grabs Thug 1 by the collar and lifts him off the ground, saying, ‘I’m Batman.’ Batman knocks out both thugs effortlessly and ties them up for the police. He then disappears into the night, leaving only the sound of his cape flapping in the wind."
evaluated_scene = "King Jumbo bumbo goes out on a marika island and handles out blue coconuts. Then a conflict arises because on of the island residents did not receive enough coconuts. The negotiation lasts for 3 hours before the kind kills all the island residents"
print(evaluate_scene_v2(evaluated_scene, agent1, tokenizer))
print(evaluate_scene_v2(evaluated_scene, agent2, tokenizer))
print(evaluate_scene_v2(evaluated_scene, agent3, tokenizer))