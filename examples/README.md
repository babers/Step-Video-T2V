# Step-Video-T2V Integration Examples

This directory contains practical examples for integrating Step-Video-T2V into your custom video generation applications.

## Examples Overview

### 1. Simple Integration (`simple_integration.py`)

A straightforward example showing direct pipeline usage for basic video generation.

**Use Case**: Quick prototyping, single video generation, learning the API

**Usage:**
```bash
python examples/simple_integration.py \
    --model_dir /path/to/stepvideo-t2v \
    --prompt "A beautiful sunset over the ocean" \
    --output_path ./my_videos \
    --num_frames 204 \
    --steps 50
```

**Features:**
- Simple, clean code
- Direct pipeline access
- Customizable parameters
- Good for understanding the basics

### 2. API Client Integration (`api_client_integration.py`)

Demonstrates how to use Step-Video-T2V as a remote service via HTTP API.

**Use Case**: Distributed deployments, microservices, separating model serving from application logic

**Usage:**

First, start the API server:
```bash
python api/call_remote_server.py --model_dir /path/to/model --port 8080
```

Then run the client:
```bash
python examples/api_client_integration.py \
    --api_url http://127.0.0.1 \
    --port 8080 \
    --prompt "A rocket launching into space"
```

**Features:**
- HTTP-based communication
- Reusable client class
- Good for production architectures
- Demonstrates text encoding and VAE decoding APIs

### 3. Custom Application Integration (`custom_app_integration.py`)

A complete example of building a custom video generation application class.

**Use Case**: Production applications, batch processing, complex workflows

**Usage:**

Single video:
```bash
python examples/custom_app_integration.py \
    --model_dir /path/to/model \
    --prompt "A chef preparing a delicious meal" \
    --output_dir ./campaign_videos
```

Batch processing from file:
```bash
# Create a prompts file
cat > prompts.txt << EOF
A cat playing with yarn
A sunset over mountains
A futuristic city at night
EOF

python examples/custom_app_integration.py \
    --model_dir /path/to/model \
    --prompts_file prompts.txt \
    --output_dir ./batch_output
```

**Features:**
- Object-oriented design
- Batch processing support
- Error handling
- Resource management
- Configuration management
- Easy to extend and customize

## Prerequisites

### 1. Model Download

Download the Step-Video-T2V model:

```bash
# Main model
huggingface-cli download stepfun-ai/stepvideo-t2v --local-dir ./models/stepvideo-t2v

# Or the turbo model for faster inference
huggingface-cli download stepfun-ai/stepvideo-t2v-turbo --local-dir ./models/stepvideo-t2v-turbo
```

### 2. API Server Setup

For examples that use remote APIs, start the API server:

```bash
python api/call_remote_server.py \
    --model_dir ./models/stepvideo-t2v \
    --port 8080
```

This provides:
- Caption API: `http://127.0.0.1:8080/caption-api`
- VAE API: `http://127.0.0.1:8080/vae-api`

### 3. GPU Requirements

- Recommended: 80GB GPU memory
- Minimum: 70GB GPU memory
- CUDA-capable GPU (tested on A100, H100)

## Common Parameters

All examples support these common parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_dir` | Required | Path to model directory |
| `--prompt` | Required | Text prompt for video |
| `--num_frames` | 204 | Number of frames (68, 136, or 204) |
| `--height` | 544 | Video height in pixels |
| `--width` | 992 | Video width in pixels |
| `--steps` | 50 | Inference steps (30-50 for main, 10-15 for turbo) |
| `--cfg_scale` | 9.0 | Guidance scale (9.0 for main, 5.0 for turbo) |
| `--time_shift` | 13.0 | Time shift (13.0 for main, 17.0 for turbo) |
| `--seed` | 1234 | Random seed for reproducibility |

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/stepfun-ai/Step-Video-T2V.git
   cd Step-Video-T2V
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   pip install flash-attn --no-build-isolation  # optional, for better performance
   ```

3. **Download the model:**
   ```bash
   huggingface-cli download stepfun-ai/stepvideo-t2v --local-dir ./models/stepvideo-t2v
   ```

4. **Start the API server (in a separate terminal):**
   ```bash
   python api/call_remote_server.py --model_dir ./models/stepvideo-t2v --port 8080
   ```

5. **Run an example:**
   ```bash
   python examples/simple_integration.py \
       --model_dir ./models/stepvideo-t2v \
       --prompt "A beautiful landscape with mountains and lakes"
   ```

## Integration Patterns

### Pattern 1: Direct Integration
```python
from stepvideo.diffusion.video_pipeline import StepVideoPipeline

pipeline = StepVideoPipeline.from_pretrained("./models/stepvideo-t2v")
video = pipeline(prompt="Your prompt", num_frames=204)
```

### Pattern 2: Application Wrapper
```python
from examples.custom_app_integration import VideoGenerationApp

app = VideoGenerationApp(model_dir="./models/stepvideo-t2v")
app.initialize()
result = app.generate("Your prompt")
```

### Pattern 3: API-Based
```python
from examples.api_client_integration import StepVideoAPIClient

client = StepVideoAPIClient("http://127.0.0.1", 8080)
embeddings = client.encode_prompt(["Your prompt"])
```

## Customization Examples

### Custom Resolution
```python
video = pipeline(
    prompt="Your prompt",
    height=768,
    width=768,
    num_frames=136
)
```

### Fast Generation (Turbo Model)
```python
video = pipeline(
    prompt="Your prompt",
    num_inference_steps=15,  # Fewer steps
    guidance_scale=5.0,      # Lower guidance
    time_shift=17.0          # Turbo setting
)
```

### High Quality
```python
video = pipeline(
    prompt="Your prompt",
    num_inference_steps=100,  # More steps
    guidance_scale=12.0       # Higher guidance
)
```

## Extending the Examples

### Add Custom Post-Processing
```python
def postprocess_video(video):
    # Your custom processing
    return processed_video

result = app.generate("Your prompt")
processed = postprocess_video(result['video'])
```

### Add Custom Prompts Enhancement
```python
def enhance_prompt(prompt):
    # Add quality modifiers
    return f"{prompt}, 4K, high quality, professional"

enhanced = enhance_prompt("A sunset")
video = pipeline(prompt=enhanced)
```

### Add Progress Tracking
```python
class TrackedVideoApp(VideoGenerationApp):
    def generate(self, prompt, **kwargs):
        self.on_start(prompt)
        result = super().generate(prompt, **kwargs)
        self.on_complete(result)
        return result
    
    def on_start(self, prompt):
        print(f"Starting generation: {prompt}")
    
    def on_complete(self, result):
        print(f"Completed: {result['output_path']}")
```

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce resolution or frame count
```bash
python examples/simple_integration.py \
    --model_dir /path/to/model \
    --prompt "Your prompt" \
    --num_frames 136 \
    --height 544 \
    --width 768
```

### Issue: API Connection Failed
**Solution**: Verify API server is running
```bash
# Check if server is running
curl http://127.0.0.1:8080/caption-api

# Restart server if needed
python api/call_remote_server.py --model_dir /path/to/model --port 8080
```

### Issue: Slow Generation
**Solution**: Use the turbo model or reduce steps
```bash
python examples/simple_integration.py \
    --model_dir ./models/stepvideo-t2v-turbo \
    --prompt "Your prompt" \
    --steps 15 \
    --cfg_scale 5.0 \
    --time_shift 17.0
```

## Additional Resources

- **Full Integration Guide**: See [INTEGRATION.md](../INTEGRATION.md) for comprehensive documentation
- **API Documentation**: See [api/call_remote_server.py](../api/call_remote_server.py) for server details
- **Main README**: See [README.md](../README.md) for project overview
- **Technical Report**: [ArXiv Paper](https://arxiv.org/abs/2502.10248)

## Contributing

If you create useful integration examples, please consider contributing them back to the project!

## Support

For issues or questions:
- GitHub Issues: [Step-Video-T2V Issues](https://github.com/stepfun-ai/Step-Video-T2V/issues)
- Discussions: Check the GitHub Discussions tab
