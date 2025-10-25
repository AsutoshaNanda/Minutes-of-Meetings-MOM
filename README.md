<div align="center">

# 📝 Minutes of Meeting (MOM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://platform.openai.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

**Automated meeting transcription and minutes generation using OpenAI Whisper and Llama 3.2**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture)

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

Minutes of Meeting (MOM) is an automated tool that converts meeting audio recordings into structured meeting minutes. The system uses OpenAI's Whisper API for accurate speech-to-text transcription and Meta's Llama 3.2-3B-Instruct model for generating well-formatted meeting minutes with summaries, discussion points, takeaways, and action items.

Built as a Google Colab notebook, this project demonstrates practical integration of multiple AI models for a real-world business automation use case.

## ✨ Features

- 🎙️ **Audio Transcription**: OpenAI Whisper API (gpt-4o-mini-transcribe) for accurate speech-to-text
- 🤖 **AI-Generated Minutes**: Llama 3.2-3B-Instruct creates structured meeting documentation
- 📊 **Structured Output**: Includes summary, attendees, discussion points, takeaways, and action items
- 🚀 **4-bit Quantization**: BitsAndBytes optimization for efficient GPU memory usage
- 💬 **Streaming Output**: Real-time token generation with TextStreamer
- 🌐 **Gradio Interface**: Simple web UI for displaying generated minutes
- 📁 **Google Drive Integration**: Direct access to audio files from mounted Drive

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│        Google Colab Environment             │
│  ┌──────────────┐   ┌──────────────────┐   │
│  │ Google Drive │──▶│ Audio File (MP3) │   │
│  └──────────────┘   └────────┬─────────┘   │
└───────────────────────────────┼─────────────┘
                                │
                                ▼
              ┌──────────────────────────────┐
              │   OpenAI Whisper API         │
              │  (gpt-4o-mini-transcribe)    │
              │   Audio → Text               │
              └────────────┬─────────────────┘
                           │
                           ▼ Transcription
              ┌──────────────────────────────┐
              │   Prompt Construction        │
              │   System + User Message      │
              └────────────┬─────────────────┘
                           │
                           ▼
              ┌──────────────────────────────┐
              │   Llama 3.2-3B-Instruct      │
              │   + 4-bit Quantization       │
              │   + TextStreamer             │
              └────────────┬─────────────────┘
                           │
                           ▼ Generated Minutes
              ┌──────────────────────────────┐
              │      Gradio Interface        │
              │   Markdown Display           │
              └──────────────────────────────┘
```

## 🚀 Installation

### Install Dependencies

```bash
# Upgrade core packages
pip install -q --upgrade bitsandbytes accelerate

# Update bitsandbytes
pip install -U -q bitsandbytes
```

### Required Packages

The notebook uses:
- `transformers` - HuggingFace model loading
- `torch` - PyTorch for model inference
- `bitsandbytes` - 4-bit quantization
- `accelerate` - Model loading optimization
- `openai` - OpenAI API client
- `gradio` - Web interface
- `huggingface_hub` - HF authentication

## ⚙️ Configuration

### 1. HuggingFace Authentication

```python
from huggingface_hub import login
from google.colab import userdata

hf_token = userdata.get('HF_TOKEN_1')
login(hf_token, add_to_git_credential=True)
```

Store your HuggingFace token in Google Colab Secrets as `HF_TOKEN_1`.

### 2. OpenAI API Key

```python
from google.colab import userdata

openai_api_key = userdata.get('HF_OPENAI')
```

Store your OpenAI API key in Google Colab Secrets as `HF_OPENAI`.

### 3. Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 4. Model Configuration

```python
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
AUDIO_MODEL = 'whisper-1'
```

## 📖 Usage

### Step 1: Set Audio File Path

```python
audio_file = '/content/drive/MyDrive/Colab Notebooks/HF- Gen_AI/Audio/denver_extract.mp3'
```

### Step 2: Transcribe Audio

```python
from openai import OpenAI

openai = OpenAI(api_key=openai_api_key)

# Open and transcribe audio file (run once only)
audio_file = open(audio_file, 'rb')

transcription = openai.audio.transcriptions.create(
    model="gpt-4o-mini-transcribe",
    file=audio_file,
    response_format='text'
)
```

### Step 3: View Transcription

```python
# Display as markdown
from IPython.display import Markdown, display
display(Markdown(transcription))

# Or get JSON format
transcription_json = openai.audio.transcriptions.create(
    model='gpt-4o-mini-transcribe',
    file=audio_file,
    response_format='json'
)
```

### Step 4: Generate Meeting Minutes

```python
# Define system and user messages
system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""

user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}
"""

messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': user_prompt}
]
```

### Step 5: Load Model with Quantization

```python
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, TextStreamer
import torch

# Configure 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_dtype='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLAMA, trust_remote_code=True)
tokenizer.pod_token = tokenizer.eos_token

# Apply chat template and tokenize
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors='pt',
    add_geneartion_prompt=True
).to('cuda')

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    LLAMA,
    device_map='auto',
    quantization_config=quant_config
)
```

### Step 6: Generate with Streaming

```python
# Setup streaming output
streamer = TextStreamer(tokenizer)

# Generate minutes
outputs = model.generate(
    inputs,
    max_new_tokens=1000,
    streamer=streamer
)

# Display result
display(Markdown(tokenizer.decode(outputs[0])))
```

### Step 7: Launch Gradio Interface

```python
import gradio as gr

def result():
    return tokenizer.decode(outputs[0])

gr.Interface(
    fn=result,
    inputs=None,
    outputs=[gr.Markdown(label="MOM")],
    allow_flagging='never'
).launch()
```

## 📁 Project Structure

```
Minutes-of-Minutes-MOM-/
│
├── 📓 minutes_of_meeting_(mom).py    # Main implementation
├── 📄 README.md                      # This file
└── 📜 LICENSE                        # Apache 2.0 License
```

## 🔬 How It Works

### 1. Audio Transcription

The system uses OpenAI's Whisper API (via `gpt-4o-mini-transcribe` model) to convert audio to text:

```python
transcription = openai.audio.transcriptions.create(
    model="gpt-4o-mini-transcribe",
    file=audio_file,
    response_format='text'  # or 'json'
)
```

### 2. Prompt Engineering

Two-part prompt structure:

**System Message:**
```
You produce minutes of meetings from transcripts, with summary, 
key discussion points, takeaways and action items with owners, 
in markdown format without code blocks.
```

**User Prompt:**
```
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription: {transcription}
```

### 3. Model Quantization

4-bit quantization configuration for memory efficiency:

```python
BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit
    bnb_4bit_use_double_quant=True,       # Double quantization
    bnb_4bit_quant_dtype='nf4',           # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16 # Compute dtype
)
```

Benefits:
- ~4x memory reduction
- Faster inference on limited GPU
- Minimal quality degradation

### 4. Generation Process

1. **Tokenization**: Apply chat template to format messages
2. **Model Loading**: Load quantized Llama 3.2-3B-Instruct
3. **Generation**: Generate up to 1000 new tokens
4. **Streaming**: Display tokens in real-time with TextStreamer
5. **Output**: Decode and display as formatted markdown

## 💡 Examples

### Example Output Structure

```markdown
# Meeting Minutes

## Summary
**Date:** [Extracted from transcript]
**Location:** [Extracted from transcript]
**Attendees:** [List of participants]

## Discussion Points
1. [Topic 1]
2. [Topic 2]
3. [Topic 3]

## Key Takeaways
- [Takeaway 1]
- [Takeaway 2]

## Action Items
| Action | Owner | Deadline |
|--------|-------|----------|
| [Task] | [Person] | [Date] |
```

### Code Walkthrough

The notebook follows this execution flow:

```python
# 1. Setup
!pip install packages → import libraries → authenticate

# 2. Data Loading
mount drive → load audio file

# 3. Transcription
openai.audio.transcriptions.create() → get text

# 4. Prompt Construction
system_message + user_prompt + transcription

# 5. Model Loading
configure quantization → load tokenizer → load model

# 6. Generation
apply_chat_template() → generate() → decode()

# 7. Interface
gradio.Interface() → launch()
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 📊 Support for multiple audio formats
- 🎯 Batch processing multiple meetings
- 👥 Speaker diarization
- 🌍 Multi-language support
- 📝 Custom MOM templates
- 📧 Email integration for distribution

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

```
Copyright 2025 Asutosha Nanda

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
```

## 👤 Author

**Asutosha Nanda**

- GitHub: [@AsutoshaNanda](https://github.com/AsutoshaNanda)

## 🙏 Acknowledgments

- **OpenAI** - Whisper API for transcription
- **Meta AI** - Llama 3.2 model
- **HuggingFace** - Transformers library
- **BitsAndBytes** - Quantization library
- **Gradio Team** - Interface framework

---

<div align="center">

⭐ **Star this repository if you find it helpful!** ⭐

Made with ❤️ for automating meeting documentation

</div>
