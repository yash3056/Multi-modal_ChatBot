import torch
import intel_extension_for_pytorch as ipex;
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Define model names and device
model_name = "meta-llama/Llama-3.1-8B-Instruct"
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

# Load tokenizer and model with proper configuration
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    pad_token_id=tokenizer.pad_token_id
)
model.to(device)

class Message:
    def __init__(self, role, content, tool_calls=None):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls

def format_chat_template(system_message, messages, tools=None):
    template_data = {
        "System": system_message,
        "Tools": tools,
        "Messages": [
            {
                "Role": msg[0],  # role
                "Content": msg[1],  # content
                "ToolCalls": None  # We'll implement tool calls later if needed
            } for msg in messages
        ]
    }
    
    formatted_messages = []
    
    # Add system message if present
    if system_message or tools:
        formatted_messages.append("<|start_header_id|>system<|end_header_id|> ")
        if system_message:
            formatted_messages.append(system_message)
        if tools:
            formatted_messages.append(
                " Cutting Knowledge Date: December 2023 "
                "When you receive a tool call response, use the output to format "
                "an answer to the orginal user question. You are a helpful assistant "
                "with tool calling capabilities."
            )
        formatted_messages.append("<|eot_id|>")
    
    # Add conversation messages
    for i, msg in enumerate(template_data["Messages"]):
        is_last = i == len(template_data["Messages"]) - 1
        
        if msg["Role"] == "user":
            formatted_messages.append("<|start_header_id|>user<|end_header_id|> ")
            if tools and is_last:
                formatted_messages.append(
                    "Given the following functions, please respond with a JSON for "
                    "a function call with its proper arguments that best answers "
                    "the given prompt. Respond in the format "
                    '{"name": function name, "parameters": dictionary of argument name and its value}. '
                    "Do not use variables. "
                )
                for tool in tools or []:
                    formatted_messages.append(f"{tool} ")
                formatted_messages.append(f"Question: {msg['Content']}<|eot_id|>")
            else:
                formatted_messages.append(f"{msg['Content']}<|eot_id|>")
            
            if is_last:
                formatted_messages.append("<|start_header_id|>assistant<|end_header_id|> ")
                
        elif msg["Role"] == "assistant":
            formatted_messages.append("<|start_header_id|>assistant<|end_header_id|> ")
            if msg["ToolCalls"]:
                for tool_call in msg["ToolCalls"]:
                    formatted_messages.append(
                        f'{{"name": "{tool_call["Function"]["Name"]}", '
                        f'"parameters": {tool_call["Function"]["Arguments"]}}}'
                    )
            else:
                formatted_messages.append(msg["Content"])
            
            if not is_last:
                formatted_messages.append("<|eot_id|>")
                
        elif msg["Role"] == "tool":
            formatted_messages.append("<|start_header_id|>ipython<|end_header_id|> ")
            formatted_messages.append(f"{msg['Content']}<|eot_id|>")
            
            if is_last:
                formatted_messages.append("<|start_header_id|>assistant<|end_header_id|> ")
    
    return "".join(formatted_messages)

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    # Format the conversation using the custom template
    formatted_prompt = format_chat_template(
        system_message=system_message,
        messages=[("user", msg[0]) if msg[0] else ("assistant", msg[1]) for msg in history] + [("user", message)]
    )
    
    # Encode the input text
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    )
    
    # Move inputs to device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][len(input_ids[0]):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Clean up special tokens and headers
    response = response.replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
    response = response.replace("<|eot_id|>", "").replace("assistant", "").strip()
    
    return response

# Create the Gradio interface
demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(
            value="You are a helpful and friendly AI assistant. Provide clear, concise, and relevant responses.",
            label="System message"
        ),
        gr.Slider(minimum=1,maximum=4096,value=256,step=1,label="Max new tokens"),
        gr.Slider(minimum=0.1,maximum=2.0,value=0.7,step=0.1,label="Temperature"),
        gr.Slider(minimum=0.1,maximum=1.0,value=0.9,step=0.05,label="Top-p (nucleus sampling)"),
    ],
    title="Llama Chat Interface",
    description="Chat with a Llama model using a custom chat template. Adjust the parameters to control the response generation."
)

if __name__ == "__main__":
    demo.launch(share=True)