import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from prompts_json import interaction_evaluation

# Load the model and tokenizer for evaluating interactions
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define the JSON schema for interaction evaluation
json_schema_interaction_evaluation = {
    "type": "object",
    "properties": {
        "Agent A": {
            "type": "object",
            "properties": {
                "Believability": {
                    "type": "object",
                    "properties": {
                        "score": { "type": "integer", "minimum": 0, "maximum": 10 },
                        "reasoning": { "type": "string" }
                    },
                    "required": ["score", "reasoning"]
                },
                "Relationship": {
                    "type": "object",
                    "properties": {
                        "score": { "type": "integer", "minimum": -5, "maximum": 5 },
                        "reasoning": { "type": "string" }
                    },
                    "required": ["score", "reasoning"]
                },
                "Knowledge": {
                    "type": "object",
                    "properties": {
                        "score": { "type": "integer", "minimum": 0, "maximum": 10 },
                        "reasoning": { "type": "string" }
                    },
                    "required": ["score", "reasoning"]
                }
            },
            "required": ["Believability", "Relationship", "Knowledge"]
        },
        "Agent B": {
            "type": "object",
            "properties": {
                "Believability": {
                    "type": "object",
                    "properties": {
                        "score": { "type": "integer", "minimum": 0, "maximum": 10 },
                        "reasoning": { "type": "string" }
                    },
                    "required": ["score", "reasoning"]
                },
                "Relationship": {
                    "type": "object",
                    "properties": {
                        "score": { "type": "integer", "minimum": -5, "maximum": 5 },
                        "reasoning": { "type": "string" }
                    },
                    "required": ["score", "reasoning"]
                },
                "Knowledge": {
                    "type": "object",
                    "properties": {
                        "score": { "type": "integer", "minimum": 0, "maximum": 10 },
                        "reasoning": { "type": "string" }
                    },
                    "required": ["score", "reasoning"]
                }
            },
            "required": ["Believability", "Relationship", "Knowledge"]
        }
    },
    "required": ["Agent A", "Agent B"]
}

# Function to extract episodes and perform interaction evaluation
def evaluate_interactions_from_csv(csv_file):
    # Load the CSV data into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Lists to store the interaction evaluation results
    interaction_eval_a_list = []
    interaction_eval_b_list = []

    # Loop through each row in the CSV to extract episodes and perform interaction evaluation
    for index, row in df.iterrows():
        # Extract episode data
        agent_a = eval(row['episode.Agent A'])  # Make sure it's properly formatted as a list
        agent_b = eval(row['episode.Agent B'])

        # Extract necessary fields for evaluation
        character1 = row['movie_scene.character1']
        character2 = row['movie_scene.character2']
        title = row['movie_scene.title']
        scenario_description = row['scenario.description']
        goals1 = row['scenario.goals1']
        goals2 = row['scenario.goals2']

        # Perform interaction evaluation
        interaction_eval = interaction_evaluation(
            agent_a, agent_b, character1, character2, title, title, scenario_description, goals1, goals2,
            model, tokenizer, json_schema_interaction_evaluation, max_length=50
        )
        
        # Extract Agent A and Agent B evaluation results
        eval_a = interaction_eval.get('Agent A', {})
        eval_b = interaction_eval.get('Agent B', {})

        # Append the evaluation results to the corresponding lists
        interaction_eval_a_list.append(eval_a)
        interaction_eval_b_list.append(eval_b)

        print(f"Performed interaction evaluation for row {index}: {interaction_eval}")

    # Add the interaction evaluation results as new columns in the original DataFrame
    df['interaction_eval.Agent A'] = interaction_eval_a_list
    df['interaction_eval.Agent B'] = interaction_eval_b_list

    # Save the updated DataFrame back to the same CSV file
    df.to_csv(csv_file, index=False)
    print("Interaction evaluations have been added to the CSV file.")

# Call the function and evaluate interactions, updating the CSV file in-place
csv_file = './scenes_scenario_validation_final.csv'  # Path to the CSV file generated by the previous script
evaluate_interactions_from_csv(csv_file)
