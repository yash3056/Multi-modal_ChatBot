# Multi-Modal Chat Application

## Overview
This project builds a multi-modal chat application that integrates text, image, audio, and video processing. It utilizes a single AI model to understand multiple data types while generating text responses. The application is built using **Llama-3.2-11B-Vision-Instruct** as the base model, with added encoders for different modalities.

## Features
- Process text, images, audio, and video.
- Generate text responses from multi-modal inputs.
- Modular architecture for ease of extension.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/multi-modal-chat.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the chat application:
   ```bash
   python app.py
   ```
2. The model processes input from different modalities and generates text-based responses.

## Architecture
- **Text Encoder**: Llama-3.2-11B-Vision-Instruct.
- **Image Encoder**: CLIP-based.
- **Audio Encoder**: Wav2Vec-based.
- **Video Encoder**: VideoMAE-based.

These encoders work in harmony without merging them into a single model.

## How it Works
1. The system identifies the type of input (text, image, audio, video).
2. The respective encoder processes the input and extracts features.
3. The features are unified in a shared latent space for generating meaningful text responses.

## Intel AI Tools Used
- **Intel® AI Analytics Toolkit**: Used for optimized model training and inference.
- **Intel® Distribution of OpenVINO™ Toolkit**: Enabled hardware acceleration and efficient model deployment.

## Contributions
Feel free to submit issues or pull requests to help improve this project.

## License
[Apache License](LICENSE)
