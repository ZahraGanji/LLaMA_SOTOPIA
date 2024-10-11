import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from prompts_json import interaction_evaluation

# Load the model and tokenizer for evaluating interactions
# MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    )
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define the JSON schema for interaction evaluation
json_schema_interaction_evaluation = {
    "type": "object",
    "properties": {
        "Agent A": {
            "type": "object",
            "properties": {
                "Believability": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Relationship": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Knowledge": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Secret": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Social Rules": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Financial and Material Benefits": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Goal": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]}
            },
            "required": ["Believability", "Relationship", "Knowledge", "Secret", "Social Rules", "Financial and Material Benefits", "Goal"]
        },
        "Agent B": {
            "type": "object",
            "properties": {
                "Believability": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Relationship": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Knowledge": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Secret": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Social Rules": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Financial and Material Benefits": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Goal": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]}
            },
            "required": ["Believability", "Relationship", "Knowledge", "Secret", "Social Rules", "Financial and Material Benefits", "Goal"]
        }
    },
    "required": ["Agent A", "Agent B"]
}

# Function to calculate the average score for each agent
def calculate_average_score(agent_eval):
    scores = []
    for category in agent_eval:
        try:
            scores.append(float(agent_eval[category]["score"]))
        except ValueError:
            pass  # If the score is not a valid number, skip it
    return sum(scores) / len(scores) if scores else 0

# Function to extract episodes and perform interaction evaluation
def evaluate_interactions_from_csv(csv_file):
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Lists to store the interaction evaluation results
    interaction_eval_a_list = []
    interaction_eval_b_list = []
    avg_score_a_list = []
    avg_score_b_list = []

    # Loop through each row in the CSV to extract episodes and perform interaction evaluation
    for index, row in df.iterrows():
        # Extract necessary fields for evaluation
        character1 = row['movie_scene.character1']
        character2 = row['movie_scene.character2']
        title = row['movie_scene.name']
        scenario_description = row['movie_scene.scenario']
        goals1 = row['movie_scene.goal1']
        goals2 = row['movie_scene.goal2']
        relationship = row['movie_scene.relationship']
        social_interaction = row['social_interaction']
        
        # Perform interaction evaluation
        interaction_eval = interaction_evaluation(
            social_interaction, character1, character2, title, title, scenario_description, goals1, goals2, relationship,
            model, tokenizer, json_schema_interaction_evaluation, max_length=50
        )
        
        # Extract Agent A and Agent B evaluation results
        eval_a = interaction_eval.get('Agent A', {})
        eval_b = interaction_eval.get('Agent B', {})

        # Calculate average scores for Agent A and Agent B
        avg_score_a = calculate_average_score(eval_a)
        avg_score_b = calculate_average_score(eval_b)

        # Append the evaluation results to the corresponding lists
        interaction_eval_a_list.append(eval_a)
        interaction_eval_b_list.append(eval_b)
        avg_score_a_list.append(avg_score_a)
        avg_score_b_list.append(avg_score_b)

        print(f"Performed interaction evaluation for row {index}: {interaction_eval}")

    # Add the interaction evaluation results and average scores as new columns in the original DataFrame
    df['interaction_eval.Agent A'] = interaction_eval_a_list
    df['interaction_eval.Agent B'] = interaction_eval_b_list
    df['overall_score.Agent A'] = avg_score_a_list
    df['overall_score.Agent B'] = avg_score_b_list

    # Save the updated DataFrame back to the same CSV file
    df.to_csv(csv_file, index=False)
    print("Interaction evaluations and average scores have been added to the CSV file.")

# Call the function and evaluate interactions, updating the CSV file in-place
csv_file = './pipeline_llama_new_70B.csv'  # Path to the CSV file generated by the previous script
evaluate_interactions_from_csv(csv_file)
