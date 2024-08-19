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
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "A detailed description of a specific movie scene illustrating the 'exchange' type of social interaction."
                },
                "characters_involved": {
                    "type": "string",
                    "description": "The characters involved in the movie scene."
                },
                "setting": {
                    "type": "string",
                    "description": "The setting of the movie scene."
                },
                "conflict_nature": {
                    "type": "string",
                    "description": "The nature of the conflict or interaction in the scene."
                }
            },
            "required": ["description", "characters_involved", "setting", "conflict_nature"]
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
                },
                "interaction": {
                    "type": "string",
                    "description": "The detailed interaction between the agents."
                }
            },
            "required": ["context", "goal", "constraints", "interaction"]
        }
    },
    "required": ["movie_scene", "scenario"]
}

# Step 5: Define your prompt
prompt = """
### Instruction ###
Please answer the following questions in a structured JSON format. Provide detailed responses to ensure comprehensive coverage of each aspect.

### Question 1 ###
Identify a movie scene that demonstrates the "exchange" type of human social interaction from the following categories: exchange, competition, cooperation, conflict, and coercion. Describe the scene in detail, including the characters involved, the nature of the interaction, and the setting.

### Question 2 ###
Based on the identified movie scene, generate a scenario of human interactions, including the context, goal, constraints, and detailed interaction between the agents. Ensure that the scenario reflects the essence of the 'exchange' interaction type.
"""

# Step 6: Create a Jsonformer instance
jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)

# Step 7: Generate the JSON data
generated_data = jsonformer()

# Step 8: Print the generated JSON
print(generated_data)
