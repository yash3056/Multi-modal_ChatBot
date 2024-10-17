import torch
from transformers import AutoProcessor, AutoTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from PIL import Image
import os

class MultimodalSystem:
    def __init__(self, llama_model="meta-llama/Llama-3.2-3B-Instruct", florence_model="microsoft/Florence-2-large"):
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        
        # Initialize Llama
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model).to(self.device)

        # Initialize Florence
        self.florence_processor = AutoProcessor.from_pretrained(florence_model, trust_remote_code=True)
        self.florence_model = AutoModelForCausalLM.from_pretrained(florence_model, trust_remote_code=True).to(self.device)

    def process_image(self, image_file):
        # Load and process the image
        image = Image.open(image_file).convert("RGB")
        inputs = self.florence_processor(images=image, return_tensors="pt").to(self.device)
        
        # Set system message
        system_message = "Your task is to describe the Image so that main model can understand the content of data."
        
        # Generate text from image
        with torch.no_grad():
            outputs = self.florence_model.generate(**inputs)
        
        # Decode the generated text
        generated_text = self.florence_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        return f"{system_message} {generated_text}"

    def process_with_llama(self, text, min_length=64, max_length=2048):
        # Prepare input for Llama
        inputs = self.llama_tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate response with Llama in chunks
        response = ""
        while len(response) < max_length:
            outputs = self.llama_model.generate(
                **inputs, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=True
            )
            chunk = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
            response += chunk
            if len(chunk) < max_length:
                break
            inputs = self.llama_tokenizer(response, return_tensors="pt").to(self.device)
        
        return response

    def process_input_and_respond(self, input_file, prompt=None):
        file_extension = os.path.splitext(input_file)[1].lower()
        
        if file_extension in ['.jpg', '.jpeg', '.png']:
            # Process image
            image_description = self.process_image(input_file)
            input_type = "image"
            if prompt is None:
                prompt = "Describe the following image content: "
            llama_input = prompt + image_description
        else:
            raise ValueError("Unsupported file type")
        
        # Process with Llama
        response = self.process_with_llama(llama_input)
        
        return {"input_type": input_type, "processed_content": image_description, "response": response}

# Usage example
system = MultimodalSystem()
result = system.process_input_and_respond("cat.jpg")
print("Input type:", result["input_type"])
print("Processed content:", result["processed_content"])
print("Llama Response:", result["response"])
