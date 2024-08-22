from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the pre-trained model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Define your prompts
prompt_1 = """
Human social interaction types are from the following categories: exchange, competition, cooperation, conflict, and coercion. 
Please provide a detailed description of a movie scene that demonstrates the "exchange" type of human social interaction. 
Include the following details:
- A description of the scene
- The characters involved
- The setting
- The nature of the conflict or interaction
"""

prompt_2 = """
Generate a scenario of human interactions based on the given movie scene and relationship constraints between characters. Include the context, goal, and constraints for the scenario. When generating scenarios, substitute the names of movie characters with more general terms such as "agents" to ensure broader applicability.
"""


def generate_response(prompt):
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response from the model
    outputs = model.generate(
        inputs["input_ids"],
        max_length=2000,  
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 3: Get responses for each prompt
response_1 = generate_response(prompt_1)
response_2 = generate_response(prompt_2)

# Step 4: Print the responses
print("Response to Prompt 1:\n", response_1)
print("\nResponse to Prompt 2:\n", response_2)
