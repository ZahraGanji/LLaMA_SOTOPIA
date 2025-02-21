# evaluation.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
import os
from tqdm import tqdm
from evaluation import evaluate_scene_v2

# Clear GPU memory and set environment variable
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Configurations
Model_judge = ["meta-llama/Meta-Llama-3.1-8B-Instruct", "databricks/dolly-v2-12b", "meta-llama/Meta-Llama-3-8B-Instruct"]
judges_no = 3

# Define evaluation schema
json_schema_evaluation = {
    "type": "object",
    "properties": {
        "is_real": {"type": "boolean"},
        "explanation": {"type": "string"}
    }
}

# Load all judge models and their corresponding tokenizers
judge_models = []
judge_tokenizers = []
for judge_model_name in Model_judge:
    judge_model = AutoModelForCausalLM.from_pretrained(judge_model_name, device_map="auto", torch_dtype=torch.float16)
    judge_model.config.use_cache = False
    judge_tokenizer = AutoTokenizer.from_pretrained(judge_model_name)
    judge_models.append(judge_model)
    judge_tokenizers.append(judge_tokenizer)

# Load scenes and scenarios from CSV
df = pd.read_csv('./pipeline_llama_new_70B.csv')


# Iterate through scenes and evaluate them
evaluation_results = []
for index, row in tqdm(df.iterrows(), total=len(df)):
    yes_count = 0
    
    for j in range(judges_no):
        answer = evaluate_scene_v2(row["movie_scene.description"], row["movie_scene.name"], 
                                   judge_models[j], judge_tokenizers[j], json_schema_evaluation)
        if answer["is_real"]:
            yes_count += 1

    validation_result = "yes" if yes_count >= 2 else "no"
    print(f"Validation result for row {index}: {validation_result}")

    # Append the validation result
    evaluation_results.append(validation_result)

# Add evaluation results to DataFrame
df['validation'] = evaluation_results

# Save updated DataFrame back to CSV
df.to_csv("./pipeline_llama_new_70B.csv", index=False)

print("Finished evaluating scenes.")
