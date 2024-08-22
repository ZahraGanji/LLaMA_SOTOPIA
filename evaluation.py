from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# source ~/NLP/venv/bin/activate
#  source ~/localdisk/pythonvenv/bin/activate
# Define a label mapping
label_mapping = {
    'LABEL_0': 'fake',
    'LABEL_1': 'real'
}

def evaluate_scene(description):
    # Context information
    context = "You will evaluate whether a given movie scene description is real or made up. If it is not real, your output will be 'fake'. If it's real, your output will be 'real'."
    
    # Combine context with the description
    input_text = f"{context} Description: {description}"
    
    # Use the classifier to evaluate the description
    result = classifier(input_text)
    print(result)
    label = result[0]['label']
    score = result[0]['score']
    
    # Map the model's output label to the desired label
    mapped_label = label_mapping.get(label, 'unknown')
    
    # Return the result with the mapped label
    return f"The scene is likely {mapped_label} with a confidence score of {score:.2f}."

def evaluate_scene_v2(description, pipeline):
    messages = [
    {"role": "system", "content": """You are a movie expert. You know every movie scene by heart. Your job is to check whether the input movie scene description is real or not.\
     If it's real - answer "real", if it's not, anwser "false". Answer best to your movie knowledge. Your answer must be either "It is not real" or "It is real" """},
    {"role": "user", "content": f"{description}"},
    ]
    outputs = pipeline(
    messages,
    max_new_tokens=256)
    return outputs[0]["generated_text"][-1]

pipeline1 = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.bfloat16})
pipeline2 = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.bfloat16})
pipeline3 = pipeline(
    "text-generation",
    model=model_name,
    model_kwargs={"torch_dtype": torch.bfloat16})
# Create a pipeline for text classification

evaluated_scene = "Two thugs are counting their loot after a robbery on a Gotham City rooftop. Suddenly, they hear a noise and look around nervously. Thug 1 says, ‘What was that?’ Thug 2 replies, ‘Probably just a rat. Let’s get out of here.’ Before they can move, a dark figure swoops down and lands in front of them. It’s Batman. Thug 1 asks, ‘Who are you?’ Batman grabs Thug 1 by the collar and lifts him off the ground, saying, ‘I’m Batman.’ Batman knocks out both thugs effortlessly and ties them up for the police. He then disappears into the night, leaving only the sound of his cape flapping in the wind."
print(evaluate_scene_v2(evaluated_scene, pipeline1))
print(evaluate_scene_v2(evaluated_scene, pipeline2))
print(evaluate_scene_v2(evaluated_scene, pipeline3))