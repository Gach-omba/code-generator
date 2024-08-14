import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the CodeGen model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
# sample prompt. This can come in as an api call from a user
prompt = """
Generate the complete HTML code for a basic web page with:
- A header containing the title 'My Web Page'.
- A body with:
    * A heading with the text 'Welcome to my page'
    * An unordered list with items 'Item 1', 'Item 2', and 'Item 3'
    * A paragraph with the text 'This is the content of the page.'
"""

inputs = tokenizer(prompt, return_tensors="pt")

# Set pad_token_id to the model's eos_token_id
model.config.pad_token_id = model.config.eos_token_id

# Generate HTML code
output = model.generate(
    inputs.input_ids, 
    attention_mask=inputs.attention_mask, 
    max_length=1024,  # can vary from 216 to 2048 based on the output needed
    num_beams=5, 
    early_stopping=True,
    no_repeat_ngram_size=2 
)

# Generated text
generated_html = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_html)
