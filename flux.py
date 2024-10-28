from diffusers import FluxPipeline
import torch
import intel_extension_for_pytorch as ipex;
import gradio as gr

# Load the Stable Diffusion model pipeline
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)

# Move the pipeline to GPU if available
device = "xpu" if torch.xpu.is_available() else "cpu"
pipe.to(device)

# Define the function to generate images based on a text prompt
def generate_image(prompt):
    torch.cuda.empty_cache()  # Clear memory cache before running
    # Generate the image with fewer steps for quicker results
    image = pipe(prompt, num_inference_steps=50).images[0]
    return image

# Set up the Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs="text",
    outputs="image",
    title="Stable Diffusion Image Generator",
    description="Enter a prompt to generate an image with Stable Diffusion"
)

# Launch the Gradio app
iface.launch(share=True)