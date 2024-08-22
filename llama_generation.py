from transformers import AutoModelForCausalLM, AutoTokenizer

# Step 1: Load the pre-trained model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  
# model_name = "databricks/dolly-v2-12b"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 2: Define your prompts
prompt_1 = """
### Instruction ###
Please answer the following questions in sequence, providing concise responses in text format. 
### Question 1 ###
Human social interaction types are from following categories: exchange, competition, cooperation, conflict, and coercion. Please find a movie scene that demonstrates the "exchange" type.

### Question 2 ###
Please generate scenarios and goals of human interactions based on the given movie scene and relationship constraints between characters. when creating the goals, try to find one point that both sides may not agree upon initially and need to collaboratively
resolve it. When generating scenarios, substitute the names of movie characters with more general terms such as "agents" to ensure broader applicability.
"""


def generate_response(prompt):
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response from the model
    outputs = model.generate(
        inputs["input_ids"],
        max_length=5000,  
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Step 3: Get responses for each prompt
response_1 = generate_response(prompt_1)

# Step 4: Print the responses
print("Response to Prompt:\n", response_1)
