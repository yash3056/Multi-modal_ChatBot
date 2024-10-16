import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model, GPT2LMHeadModel

# Initialize the feature extractor and model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("openai/whisper-tiny")
audio_model = Wav2Vec2Model.from_pretrained("openai/whisper-tiny", torch_dtype=torch.float16)
text_model = GPT2LMHeadModel.from_pretrained("gpt2", torch_dtype=torch.float16)

# Function to preprocess audio
def preprocess_audio(audio_chunk):
    input_values = feature_extractor(audio_chunk, sampling_rate=16000, return_tensors="pt").input_values
    return input_values

# Function to generate text from audio chunk
def generate_text_from_chunk(audio_chunk):
    audio_features = preprocess_audio(audio_chunk)
    features = audio_model(audio_features).last_hidden_state  # Ensure the shape fits Wav2Vec2Model input
    text = "summarize the content of the audio"
    text_features = text_model.encode(text, return_tensors="pt")
    
    # Fusion (simple concatenation for demonstration)
    fused_features = torch.cat((features, text_features), dim=-1)
    
    # Generate text
    outputs = text_model.generate(fused_features, max_length=100)
    return text_model.decode(outputs[0])


# Function to split audio into chunks and generate text
def generate_text(audio_file):
    info = torchaudio.info(audio_file)
    sampling_rate = info.sample_rate
    num_frames = info.num_frames
    chunk_size = 30 * sampling_rate  # 30 seconds
    num_chunks = (num_frames + chunk_size - 1) // chunk_size  # Calculate number of chunks
    
    generated_texts = []
    for i in range(num_chunks):
        start_frame = i * chunk_size
        end_frame = min((i + 1) * chunk_size, num_frames)
        audio_chunk, _ = torchaudio.load(audio_file, frame_offset=start_frame, num_frames=end_frame - start_frame)
        generated_text = generate_text_from_chunk(audio_chunk)
        generated_texts.append(generated_text)
    
    return " ".join(generated_texts)

# Example usage
audio_file = "linus-original.wav"
generated_text = generate_text(audio_file)
print(generated_text)