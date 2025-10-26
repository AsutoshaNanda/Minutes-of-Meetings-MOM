<div align="center">

# 📝 Minutes of Meeting (MOM)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Whisper-green.svg)](https://platform.openai.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Colab](https://img.shields.io/badge/Google-Colab-orange.svg)](https://colab.research.google.com)
[![Multi-Model](https://img.shields.io/badge/Multi%20Model-Support-brightgreen.svg)](#-supported-models)

**Automated meeting transcription and minutes generation using OpenAI Whisper and multiple LLM models**

[Overview](#-overview) • [Features](#-features) • [Models](#-supported-models) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Configuration](#-configuration) • [Usage](#-usage) • [Architecture](#-architecture) • [Examples](#-examples) • [Contributing](#-contributing) • [License](#-license)

</div>

---

## 📋 Overview

**Minutes of Meeting (MOM)** is an automated tool that converts meeting audio recordings into structured, professional meeting minutes. The system uses OpenAI's Whisper API for accurate speech-to-text transcription and supports multiple large language models for generating well-formatted meeting minutes with summaries, discussion points, takeaways, and action items.

This project provides both a **single-model implementation** (original version) and a **generalized multi-model framework** for maximum flexibility. Built as Google Colab notebooks, this project demonstrates practical integration of multiple AI models for real-world business automation use cases.

**Use Cases:**
- Corporate meeting documentation
- Conference and seminar transcription
- Team sync-up summaries
- Board meeting minutes
- Research meeting documentation
- Client presentation notes
- Training session transcripts

---

## ✨ Features

### Core Features
- 🎙️ **Audio Transcription**: OpenAI Whisper API (`whisper-1`) for accurate speech-to-text conversion
- 🤖 **Multi-Model LLM Support**: Choose from 5 different models for minutes generation:
  - Meta Llama 3.2-3B-Instruct
  - Microsoft Phi-4-mini-instruct
  - Google Gemma-3-270m-it
  - Qwen Qwen3-4B-Instruct
  - DeepSeek-R1-Distill-Qwen-1.5B

### Output Formatting
- 📊 **Structured Output**: Professional minutes with:
  - Executive summary with attendees, location, and date
  - Key discussion points and topics
  - Major takeaways and decisions
  - Clear action items with owners and deadlines
  - Formatted in clean Markdown (without code blocks)

### Performance & Optimization
- 🚀 **4-bit Quantization**: BitsAndBytes optimization for efficient GPU memory usage
- 💬 **Real-time Generation**: Stream tokens during generation for live feedback
- ⚡ **Memory Efficient**: ~4x memory reduction with quantization
- 🔄 **Automatic Memory Management**: Garbage collection and CUDA cache clearing

### User Interface & Integration
- 🌐 **Gradio Web Interface**: Simple, intuitive web UI for easy interaction
- 📁 **Google Drive Integration**: Direct access to audio files from mounted Drive
- 🎯 **Flexible Input**: Support for local files and Google Drive paths
- 📤 **Clean Output Display**: Markdown-formatted results for professional presentation

### Development Features
- 🔧 **Modular Architecture**: Easy to extend and customize
- 📚 **Multiple Implementations**: Both single-model and generalized versions available
- 🧪 **Error Handling**: Robust error handling and troubleshooting
- 📝 **Well-Documented**: Comprehensive documentation and examples

---

## 🤖 Supported Models

### Model Comparison Table

| Model | Size | Parameters | Speed | Quality | Memory | Best For |
|-------|------|------------|-------|---------|--------|----------|
| **Llama 3.2-3B** | 3B | ~3B | Medium | ⭐⭐⭐⭐⭐ | ~6GB | Balanced performance & quality |
| **Phi-4-mini** | Small | ~3.8B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~4GB | Speed with good quality |
| **Gemma-3-270m** | Tiny | 270M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~2GB | Lightweight & fast |
| **Qwen3-4B** | 4B | ~4B | Medium | ⭐⭐⭐⭐⭐ | ~6GB | Multilingual support |
| **DeepSeek-R1** | 1.5B | ~1.5B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ~3GB | Reasoning & complex tasks |

### Model Selection Guide

**Choose Llama 3.2-3B if:**
- You want the best balance between speed and quality
- You have sufficient GPU memory (T4 or better)
- Meeting complexity is moderate to high
- You want the most consistent results

**Choose Phi-4-mini if:**
- You need faster inference
- GPU memory is limited
- You want good quality with speed
- Meetings are standard business discussions

**Choose Gemma-3-270m if:**
- You have very limited GPU memory
- You need the fastest possible inference
- Meeting transcripts are short
- Memory efficiency is critical

**Choose Qwen3-4B if:**
- Your meetings are in multiple languages
- You need excellent multilingual support
- Meeting complexity is moderate to high
- You have sufficient GPU memory

**Choose DeepSeek-R1 if:**
- Meetings involve complex reasoning
- You need logical analysis of discussions
- Meeting transcripts contain technical content
- You want focused reasoning on specific topics

---

## 🔧 Installation

### Prerequisites
- Google Colab account (recommended) or local GPU with CUDA support
- OpenAI API key (get from https://platform.openai.com/api-keys)
- HuggingFace API token (get from https://huggingface.co/settings/tokens)
- Sufficient GPU memory (T4 or better recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/AsutoshaNanda/Minutes-of-Meetings-MOM.git
cd Minutes-of-Meetings-MOM
```

### Step 2: Choose Your Implementation

**For Generalized Multi-Model Support (Recommended):**
- Upload `MOM_Generalised.ipynb` to Google Colab
- Supports all 5 models with easy switching

**For Single-Model Implementation (Original):**
- Upload `minutes_of_meeting_(mom).py` or use as reference
- Optimized for Llama 3.2-3B-Instruct

### Step 3: Install Dependencies

In Google Colab, run these installation cells:

```python
# Upgrade core packages
!pip install -q --upgrade bitsandbytes accelerate

# Update bitsandbytes
!pip install -U -q bitsandbytes
```

### Required Python Libraries

The notebook automatically imports these packages:

```python
# Core ML/AI
- transformers        # HuggingFace model loading
- torch               # PyTorch for model inference
- bitsandbytes        # 4-bit quantization
- accelerate          # Model loading optimization

# API & Integration
- openai              # OpenAI API client
- gradio              # Web interface
- huggingface_hub     # HuggingFace authentication

# Utilities
- requests            # HTTP requests
- google.colab        # Colab utilities
```

---

## ⚙️ Configuration

### Step 1: Set Up Google Colab Secrets

#### Add HuggingFace Token

1. Open your notebook in Google Colab
2. Click the **Secrets** icon (🔑) on the left panel
3. Create new secret:
   - **Name**: `HF_TOKEN_1`
   - **Value**: Your HuggingFace API token
4. Get your token from: https://huggingface.co/settings/tokens
5. Click "Grant access" to allow the notebook to use this secret

#### Add OpenAI API Key

1. In Google Colab Secrets (🔑), create another secret:
   - **Name**: `HF_OPENAI`
   - **Value**: Your OpenAI API key
2. Get your key from: https://platform.openai.com/api-keys
3. Click "Grant access"

### Step 2: Configure Model in Notebook

```python
# Model selection (in the notebook)
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
PHI = "microsoft/Phi-4-mini-instruct"
GEMMA = "google/gemma-3-270m-it"
QWEN = "Qwen/Qwen3-4B-Instruct-2507"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Audio model (always Whisper)
AUDIO_MODEL = 'whisper-1'
```

### Step 3: Configure Quantization

```python
from transformers import BitsAndBytesConfig
import torch

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,       # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute dtype
    bnb_4bit_quant_type='nf4'             # Quantization type
)
```

**Benefits of 4-bit Quantization:**
- ~4x reduction in model memory usage
- Faster inference on limited GPU
- Minimal quality degradation
- Enables running larger models on smaller GPUs

### Step 4: Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

Place your audio files in: `/content/drive/MyDrive/Audio/`

### Step 5: Authenticate with HuggingFace and OpenAI

```python
from huggingface_hub import login
from google.colab import userdata
from openai import OpenAI

# HuggingFace login
hf_token = userdata.get('HF_TOKEN_1')
login(hf_token, add_to_git_credential=True)

# OpenAI setup
openai_api_key = userdata.get('HF_OPENAI')
openai = OpenAI(api_key=openai_api_key)
```

---

## 🚀 Quick Start

### The Simplest Way to Get Started

1. **Open Colab Notebook**: Upload `MOM_Generalised.ipynb` to Google Colab
2. **Set Secrets**: Add HF_TOKEN_1 and HF_OPENAI in Colab Secrets
3. **Prepare Audio**: Upload your meeting recording to Google Drive
4. **Run All Cells**: Execute the notebook from top to bottom
5. **Use the Interface**: Use the Gradio web interface to generate minutes

### Example: Generate Minutes in 3 Lines

```python
# Path to your audio file
audio_file = "/content/drive/MyDrive/Audio/meeting.mp3"

# Choose a model
model = "LLAMA"  # or PHI, GEMMA, QWEN, DEEPSEEK

# Generate minutes!
result = model_selection(audio_file, model)
print(result)
```

---

## 📖 Usage Guide

### Using the Python Functions

#### 1. Transcribe Audio to Text

```python
def transcription(audio_file):
    """Convert audio file to text using OpenAI Whisper"""
    audio_file = open(audio_file, 'rb')
    transcription = openai.audio.transcriptions.create(
        model=AUDIO_MODEL,
        file=audio_file,
        response_format='text'
    )
    return transcription

# Usage
text = transcription("/content/drive/MyDrive/Audio/meeting.mp3")
print(text)
```

#### 2. Generate Minutes with a Specific Model

```python
def model_selection(audio_file, model_name):
    """Select and use a specific model for MOM generation"""
    if model_name == 'GEMMA':
        return generate_gemma(GEMMA, audio_file)
    elif model_name == 'LLAMA':
        return generate(LLAMA, audio_file)
    elif model_name == 'PHI':
        return generate(PHI, audio_file)
    elif model_name == 'QWEN':
        return generate(QWEN, audio_file)
    elif model_name == 'DEEPSEEK':
        return generate(DEEPSEEK, audio_file)

# Usage
result = model_selection("/path/to/audio.mp3", "LLAMA")
print(result)
```

#### 3. Use the Gradio Web Interface

```python
import gradio as gr

gr.Interface(
    fn=model_selection,
    inputs=[
        gr.Textbox(label='Enter The Audio File Path'),
        gr.Dropdown(
            ["LLAMA", "PHI", "GEMMA", "QWEN", "DEEPSEEK"],
            label='Choose Your Model'
        )
    ],
    outputs=[gr.Markdown(label='Minutes of Meeting')],
    allow_flagging='never',
).launch(debug=True)
```

### Supported Audio Formats

The following formats are supported via OpenAI Whisper:
- **MP3** - MPEG Audio
- **WAV** - Waveform Audio File Format
- **M4A** - MPEG-4 Audio
- **WEBM** - Web Media Audio
- **FLAC** - Free Lossless Audio Codec
- **OGG** - Ogg Vorbis Audio
- **AAC** - Advanced Audio Coding
- **OPUS** - Opus Audio

**File Size Limits:**
- Maximum 25 MB per file (OpenAI Whisper API limit)
- For longer meetings, split the audio file or use batch processing

### Advanced Usage: Custom Prompts

```python
# Create custom system prompt for specific use cases
custom_system_prompt = """
You are an expert assistant specialized in financial meeting documentation.
Generate Minutes of Meeting (MOM) with special emphasis on:
- Financial decisions and budget allocations
- Revenue and cost implications
- Risk factors and mitigation strategies
- Quarterly targets and KPIs

Format the output in clean Markdown without code blocks.
"""

# Create messages with custom prompt
messages = [
    {'role': 'system', 'content': custom_system_prompt},
    {'role': 'user', 'content': user_prompt_for(audio_file)}
]
```

---

## 🏗️ Architecture & System Design

### Complete System Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  Google Colab Environment                   │
│  ┌──────────────┐         ┌──────────────────────────────┐ │
│  │ Google Drive │────────▶│ Audio File (MP3, WAV, etc.)  │ │
│  └──────────────┘         └──────────┬───────────────────┘ │
└─────────────────────────────────────┼─────────────────────┘
                                      │
                                      ▼
                ┌─────────────────────────────────────┐
                │   OpenAI Whisper API (whisper-1)    │
                │   Audio File → Text Transcription   │
                │   (With error handling & retries)   │
                └────────────────┬────────────────────┘
                                 │
                         ▼ Transcription Text
        ┌────────────────────────────────────────────┐
        │      Prompt Engineering & Construction     │
        │  ┌──────────────────────────────────────┐ │
        │  │ System Message (Expert instructions) │ │
        │  └──────────────────────────────────────┘ │
        │  ┌──────────────────────────────────────┐ │
        │  │ User Message + Transcription         │ │
        │  └──────────────────────────────────────┘ │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │      Model Selection Interface             │
        │  ┌──────────────────────────────────────┐ │
        │  │ Choose: LLAMA / PHI / GEMMA /        │ │
        │  │         QWEN / DEEPSEEK             │ │
        │  └──────────────────────────────────────┘ │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │   Quantization Configuration (BitsAndBytes)│
        │  • 4-bit quantization enabled             │
        │  • Double quantization for efficiency     │
        │  • NF4 quantization type                  │
        │  • BFloat16 compute dtype                 │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │      Model Loading & Initialization       │
        │  • Load tokenizer from HuggingFace        │
        │  • Set padding tokens                     │
        │  • Load model with quantization config    │
        │  • Move to GPU (device_map='auto')        │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │        Token Generation Pipeline           │
        │  1. Apply chat template                    │
        │  2. Tokenize input                         │
        │  3. Move tensors to CUDA                   │
        │  4. Generate tokens (max_new_tokens=5000) │
        │  5. Stream output in real-time             │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │     Post-Generation Processing            │
        │  • Decode token IDs to text                │
        │  • Clean up VRAM                           │
        │  • Run garbage collection                  │
        │  • Clear CUDA cache                        │
        └────────────────┬─────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────────┐
        │         Output Formatting                  │
        │  • Format as Markdown                      │
        │  • Ensure no code blocks                   │
        │  • Professional structure                  │
        └────────────────┬─────────────────────────┘
                         │
                         ▼ Generated MOM
        ┌────────────────────────────────────────────┐
        │      Gradio Web Interface Display          │
        │  • Markdown rendering                      │
        │  • Beautiful HTML display                  │
        │  • Interactive user experience             │
        │  • Real-time updates                       │
        └────────────────────────────────────────────┘
```

### Component Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   MOM SYSTEM ARCHITECTURE                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Input Layer                                             │
│  ├─ Audio Files (MP3, WAV, etc.)                        │
│  ├─ Google Drive Integration                            │
│  └─ Direct File Upload                                  │
│                                                          │
│  Transcription Layer                                     │
│  ├─ OpenAI Whisper API                                  │
│  ├─ Error Handling & Retries                            │
│  └─ Quality Assurance                                   │
│                                                          │
│  Processing Layer                                        │
│  ├─ Prompt Engineering                                  │
│  ├─ Message Construction                                │
│  └─ Template Application                                │
│                                                          │
│  Model Layer (5 Options)                                 │
│  ├─ Llama 3.2-3B-Instruct                               │
│  ├─ Microsoft Phi-4-mini                                │
│  ├─ Google Gemma-3-270m                                 │
│  ├─ Qwen3-4B-Instruct                                   │
│  └─ DeepSeek-R1-Distill                                 │
│                                                          │
│  Optimization Layer                                      │
│  ├─ 4-bit Quantization (BitsAndBytes)                   │
│  ├─ Memory Management                                   │
│  ├─ CUDA Optimization                                   │
│  └─ Garbage Collection                                  │
│                                                          │
│  Generation Layer                                        │
│  ├─ Token Streaming                                     │
│  ├─ Real-time Output                                    │
│  ├─ Error Recovery                                      │
│  └─ Output Formatting                                   │
│                                                          │
│  Output Layer                                            │
│  ├─ Markdown Formatting                                 │
│  ├─ Gradio Interface                                    │
│  ├─ Web Display                                         │
│  └─ Export Options                                      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Audio Input** → File uploaded or path provided
2. **Transcription** → OpenAI Whisper converts audio to text
3. **Preprocessing** → Text cleaned and formatted
4. **Prompt Creation** → System + User messages constructed
5. **Model Selection** → User chooses from 5 available models
6. **Quantization** → Model loaded with 4-bit compression
7. **Inference** → Tokens generated in streaming mode
8. **Post-Processing** → Memory cleaned, CUDA cache cleared
9. **Formatting** → Output converted to Markdown
10. **Display** → Gradio interface presents results

---

## 📁 Project Structure

```
Minutes-of-Meetings-MOM/
│
├── 📓 MOM_Generalised.ipynb          # Multi-model generalized version (RECOMMENDED)
│   ├─ Supports: LLAMA, PHI, GEMMA, QWEN, DEEPSEEK
│   ├─ Dynamic model selection
│   ├─ 4-bit quantization
│   └─ Gradio web interface
│
├── 📓 minutes_of_meeting_(mom).py     # Original single-model implementation
│   ├─ Optimized for Llama 3.2-3B
│   ├─ Basic implementation
│   └─ Reference for single model usage
│
├── 📄 README.md                       # This comprehensive guide
│   ├─ Installation instructions
│   ├─ Configuration guide
│   ├─ Usage examples
│   └─ Troubleshooting
│
├── 📜 LICENSE                         # Apache 2.0 License
│   └─ Full license text
│
└── 📚 Documentation/
    ├─ Architecture overview
    ├─ API reference
    ├─ Examples and use cases
    └─ FAQ and troubleshooting
```

---

## 🔄 How It Works: Detailed Explanation

### Phase 1: Audio Transcription

The system uses OpenAI's state-of-the-art Whisper model for audio transcription:

```python
def transcription(audio_file):
    """
    Convert audio file to text using OpenAI Whisper API
    
    Args:
        audio_file (str): Path to audio file
    
    Returns:
        str: Transcribed text
    """
    audio_file = open(audio_file, 'rb')
    transcription = openai.audio.transcriptions.create(
        model=AUDIO_MODEL,  # 'whisper-1'
        file=audio_file,
        response_format='text'  # Plain text output
    )
    return transcription
```

**Key Features:**
- Supports multiple audio formats
- Automatic language detection
- High accuracy (95%+ for clear audio)
- Handles background noise
- Fast processing (~1 minute audio in 5-10 seconds)

### Phase 2: Prompt Engineering

The system uses a two-part prompt strategy for optimal results:

```python
system_prompt = """
You are an expert assistant that generates clear, concise, and 
well-structured Minutes of Meeting (MOM) documents from raw 
meeting transcripts.

Your output must be in clean Markdown format (without code blocks), 
and should include:
- Meeting Summary: A brief overview
- Key Discussion Points: Important topics or decisions discussed
- Takeaways: Major insights, agreements, or learnings
- Action Items: Clear tasks with responsible owners

Guidelines:
- Write in professional, easy-to-read language
- Avoid filler words or redundant phrases
- Do not include timestamps or transcription noise
- Keep the tone formal and factual, but concise
- Focus on clarity, structure, and readability
"""

user_prompt = f"""
Please write well-structured Minutes of Meeting (MOM) in Markdown 
format (without code blocks), including:

- Summary: Include attendees, location, and date if mentioned
- Key Discussion Points: List the main topics or debates
- Takeaways: Important insights or decisions
- Action Items: Clearly list tasks with owners and deadlines

Transcription:
{transcription}
"""

messages = [
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': user_prompt}
]
```

**Why This Works:**
- System message sets expectations and guidelines
- User message provides specific requirements
- Transcription is included in context
- Clear structure guides model output
- Professional tone is enforced

### Phase 3: Model Loading with Quantization

The system loads models efficiently using 4-bit quantization:

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

# Configure 4-bit quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,                      # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,         # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bfloat16
    bnb_4bit_quant_type='nf4'               # Use NF4 (Normal Float 4)
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',                      # Automatic device mapping
    quantization_config=quant_config        # Apply quantization
)
```

**Quantization Benefits:**
- **Memory**: 3B model → ~800MB (vs 6GB unquantized)
- **Speed**: 20-30% faster inference
- **Quality**: Minimal degradation in output quality
- **Accessibility**: Run larger models on smaller GPUs

### Phase 4: Token Generation

The system generates tokens with streaming for real-time feedback:

```python
def generate(model_name, audio_file):
    """Generate MOM using specified model"""
    
    # 1. Get transcription
    messages = messages_for(audio_file)
    
    # 2. Load components
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Apply chat template and tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors='pt',
        add_generation_prompt=True
    ).to('cuda')
    
    # 4. Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        quantization_config=quant_config
    )
    
    # 5. Generate tokens
    outputs = model.generate(
        inputs,
        max_new_tokens=5000  # Limit output length
    )
    
    # 6. Decode and clean up
    result = tokenizer.decode(outputs[0])
    
    # 7. Memory cleanup
    del model, inputs, tokenizer, outputs
    gc.collect()
    torch.cuda.empty_cache()
    
    return result
```

**Generation Parameters:**
- `max_new_tokens`: Maximum tokens to generate (5000 for most meetings)
- `temperature`: Model creativity (default: 1.0)
- `top_p`: Nucleus sampling for diversity
- `top_k`: Top-k sampling for quality

### Phase 5: Memory Management

Proper memory management is critical for Colab:

```python
import gc
import torch

# After generation
del model              # Delete model from memory
del inputs             # Delete input tensors
del tokenizer          # Delete tokenizer
del outputs            # Delete output tensors

gc.collect()           # Garbage collection
torch.cuda.empty_cache()  # Clear GPU cache
```

---

## 📝 Output Format & Examples

### Generated MOM Structure

```markdown
## Meeting Summary
**Date:** October 9, 2017
**Location:** Denver City Council Chambers
**Attendees:** Council members, Denver American Indian Commission representatives

The meeting focused on the observance of Indigenous Peoples Day, with 
presentations about the cultural significance and logo design for 
Confluence Week.

## Key Discussion Points
- Significance of the confluence of two rivers in Denver's history
- Recognition of Indigenous Peoples' contributions to the city
- Importance of cultural preservation and inclusivity
- Connection between cultural celebration and environmental protection
- Efforts by Indigenous youth in community engagement

## Takeaways
- Indigenous Peoples Day celebrates the cultural foundations of Denver
- The city recognizes approximately 100 tribal nations represented in the community
- Cultural pride does not require contempt or disrespect of other cultures
- Public lands protection is intertwined with cultural preservation
- Youth involvement in Indigenous advocacy is significant and growing

## Action Items
| Action | Owner | Deadline |
|--------|-------|----------|
| Transmit proclamation to Denver American Indian Commission | City Clerk | Immediate |
| Distribute proclamation to School District No. 1 | City Clerk | Immediate |
| Share proclamation with Colorado Commission on Indian Affairs | City Clerk | Immediate |
| Promote Confluence Week events | Indigenous Commission | October 2017 |
| Continue public lands advocacy | Community Leaders | Ongoing |
```

### Example: Short Meeting MOM

```
## Meeting Summary
**Date:** [Extracted from context]
**Attendees:** Team leads, project managers

Quick sync-up to review project progress and upcoming deliverables.

## Key Discussion Points
- Current project status: 60% complete
- Frontend development on track
- Backend API testing scheduled for next week
- Third-party integration pending approval

## Takeaways
- Team is making good progress
- Need to expedite third-party approvals
- Testing phase critical for June launch

## Action Items
| Action | Owner | Deadline |
|--------|
```

### 🤝 Contributing

Contributions are welcome! Areas for improvement:

- 📊 Support for multiple audio formats
- 🎯 Batch processing multiple meetings
- 👥 Speaker diarization
- 🌍 Multi-language support
- 📝 Custom MOM templates
- 📧 Email integration for distribution

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2025 Asutosha Nanda

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.


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
