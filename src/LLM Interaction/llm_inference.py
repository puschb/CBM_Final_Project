from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from dotenv import load_dotenv
import os

load_dotenv()

class GPT2Inference:
    def __init__(self):
        # Load the tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.model.eval()  # Set model to evaluation mode
        self.pad_token_id = self.tokenizer.eos_token_id

    def generate_text(self, prompt: str, max_length: int = 50, num_return_sequences: int = 1):
        #Runs inference on GPT2 Model
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Pass attention mask
                max_length=max_length,
                pad_token_id=self.pad_token_id
            )

        # Decode and return each generated sequence
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts
    

class LLama38bInference:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", quantization_bits=None, device_map = "auto"):         
        # Check for valid quantization_bits if specified
        self.access_token = os.getenv("HF_ACCESS_TOKEN")
        if quantization_bits is not None:
            if quantization_bits not in [4, 8]:
                raise ValueError("Quantization bits must be either 4 or 8.")
            
            # Configure quantization
            quant_config = BitsAndBytesConfig(
                load_in_8bit=(quantization_bits == 8),
                load_in_4bit=(quantization_bits == 4)
            )
            
            # Load the quantized model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,  # Automatically assign model to GPU if available
                use_auth_token=self.access_token
            )
        else:
            # Load the full-precision model without quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device_map,
                use_auth_token=self.access_token
            )
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=self.access_token)

        # Set padding token to avoid warnings
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use eos_token as pad_token
        self.pad_token_id = self.tokenizer.pad_token_id  # Set padding token ID

        self.model.eval()  # Set model to evaluation mode

    def generate_text(self, prompt: str, max_length: int = 50, num_return_sequences: int = 1):
        """
        Generate text that continues from the given prompt.
        
        Args:
            prompt (str): The initial text to start generation from.
            max_length (int): The maximum length of the generated text (in tokens).
            num_return_sequences (int): The number of generated sequences to return.

        Returns:
            List of generated text sequences.
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)

        # Generate text with sampling enabled
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids.to(self.model.device),  # Move inputs to the model's device
                attention_mask=inputs.attention_mask.to(self.model.device),  # Pass attention mask
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                #do_sample=True,              # Enable sampling
                #top_k=50,                    # Limits sampling to the top-k tokens (optional)
                #top_p=0.95,                  # Uses nucleus sampling (optional)
                #temperature=0.7,             # Controls randomness (lower values make output more focused)
                pad_token_id=self.pad_token_id  # Set pad_token_id to eos_token_id
            )

        # Decode and return each generated sequence
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_texts
    
    