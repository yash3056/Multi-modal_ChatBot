import requests
import torch
import torchaudio
import torchaudio.transforms as T
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, Wav2Vec2Processor, Wav2Vec2Model, AutoImageProcessor, VideoMAEModel
from torchvision.transforms import ToTensor
from moviepy.editor import VideoFileClip
from accelerate import Accelerator

accelerator = Accelerator(device_placement=True, mixed_precision="bf16")
print(accelerator.device)

# Load Llama model for text-image processing
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
model.tie_weights()
model.to(accelerator.device)

processor = AutoProcessor.from_pretrained(model_id)
print(model.device)

# Load Wav2Vec2 processor and model for audio encoding
wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(model.device)

# Load VideoMAE processor and model for video encoding
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(model.device)

# Image input
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Load audio from a local file (WAV format)
audio_file = "linus-original.wav"
# Use the soundfile backend to load the audio
torchaudio.set_audio_backend("soundfile")
waveform, sample_rate = torchaudio.load(audio_file, backend="soundfile")

# Resample the audio to 16000 Hz
resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
waveform = resampler(waveform)

# Ensure the waveform tensor has the correct shape
if waveform.dim() == 2:
    waveform = waveform.mean(dim=0)  # Convert to mono by averaging channels if stereo
waveform = waveform.unsqueeze(0)  # Add batch dimension

# Process the waveform tensor
audio_waveform = wav2vec_processor(waveform, sampling_rate=16000, return_tensors="pt").input_values

# Move the tensor to the appropriate device and ensure correct shape
audio_waveform = audio_waveform.squeeze().unsqueeze(0).to(model.device)

# Extract audio features
audio_features = wav2vec_model(audio_waveform).last_hidden_state

# Clear cache to free up memory
torch.xpu.empty_cache()

# Video input
# video_url = "path_to_your_video.mp4"
# video_clip = VideoFileClip(video_url)
# frames = []
# for frame in video_clip.iter_frames():
#     frame = ToTensor()(Image.fromarray(frame))
#     frames.append(frame.unsqueeze(0))  # Collect frames as batch
# video_tensor = torch.cat(frames).to(model.device)
# video_features = video_model(video_tensor).last_hidden_state

# Create multimodal message
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "summarize the audio"},
        {"type": "audio"},
        #{"type": "video"}
    ]}
]

# Combine inputs for the model
combined_inputs = processor(
    images=image,
    text="summarize the audio",
    audios=audio_waveform,
    return_tensors="pt"
).to(model.device)

# Generate output
output = model.generate(**combined_inputs, max_new_tokens=30)
print(processor.decode(output[0]))