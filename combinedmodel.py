import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, LlamaForCausalLM

class MultimodalSystem:
    def __init__(self, whisper_model="openai/whisper-tiny", llama_model="meta-llama/Llama-3.2-1B-Instruct"):
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")

        # Initialize Whisper
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model).to(self.device)
        
        # Initialize Llama
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model)
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model).to(self.device)

    def transcribe_audio(self, audio_file):
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_file)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to numpy array
        waveform = waveform.squeeze().numpy()
        
        # Process audio with Whisper
        input_features = self.whisper_processor(waveform, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.device)
        
        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription

    def process_with_llama(self, text, max_length=2048):
        # Prepare input for Llama
        inputs = self.llama_tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate response with Llama
        outputs = self.llama_model.generate(**inputs, max_length=max_length)
        response = self.llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return response

    def process_audio_and_respond(self, audio_file, prompt="Summarize the following audio transcription: "):
        # Transcribe audio
        transcription = self.transcribe_audio(audio_file)
        
        # Prepare input for Llama
        llama_input = prompt + transcription
        
        # Process with Llama
        response = self.process_with_llama(llama_input)
        
        return {"transcription": transcription, "response": response}

# Usage example
system = MultimodalSystem()
result = system.process_audio_and_respond("linus-original.wav")
print("Llama Response:", result["response"])