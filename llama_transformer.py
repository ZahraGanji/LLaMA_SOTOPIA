# Step 2: Import required libraries
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 3: Load the pre-trained model and tokenizer
model_name = "databricks/dolly-v2-12b"  # Replace with your desired model
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 4: Define your JSON schema
json_schema = {
    "type": "object",
    "properties": {
        "movie_scene": {
            "type": "string",
            "description": "A detailed description of a specific movie scene illustrating the 'conflict' type of social interaction."
        },
        "scenario": {
            "type": "object",
            "properties": {
                "context": {
                    "type": "string",
                    "description": "The context of the human interaction scenario."
                },
                "goal": {
                    "type": "string",
                    "description": "The goal of the human interaction scenario."
                },
                "constraints": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Constraints or conditions related to the scenario."
                }
            },
            "required": ["context", "goal", "constraints"]
        }
    },
    "required": ["movie_scene", "scenario"]
}

# Step 5: Define your prompt
prompt = """
### Instruction ###
Please answer the following questions in sequence, providing concise responses in text format. When generating scenarios, substitute the names of movie characters with more general terms such as "agents" to ensure broader applicability.

### Question 1 ###
Provide a detailed description of a specific movie scene that illustrates the "conflict" type of human social interaction. Describe the scene in detail, including the characters involved, the nature of the conflict, and the setting.

### Question 3 ###
Generate a scenario of human interactions based on the given movie scene and relationship constraints between characters. Include the context, goal, and constraints for the scenario.
"""

# Step 6: Create a Jsonformer instance
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)

# Step 7: Generate the JSON data
generated_data = jsonformer()

# Step 8: Print the generated JSON
print(generated_data)
