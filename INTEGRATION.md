# Step-Video-T2V Integration Guide

This guide provides comprehensive instructions for integrating Step-Video-T2V into your custom video generation applications.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Integration Approaches](#integration-approaches)
4. [Quick Start Examples](#quick-start-examples)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Production Deployment](#production-deployment)
8. [Troubleshooting](#troubleshooting)

## Overview

Step-Video-T2V can be integrated into your application in several ways:

1. **Direct Pipeline Integration** - Use the Python API directly in your application
2. **Remote API Integration** - Deploy the model as a service and call it via HTTP API
3. **Distributed Multi-GPU** - Scale across multiple GPUs for production workloads

## Installation

### 1. Install the Package

```bash
git clone https://github.com/stepfun-ai/Step-Video-T2V.git
cd Step-Video-T2V
pip install -e .
```

### 2. Install Optional Dependencies

For better performance, install flash-attention:

```bash
pip install flash-attn --no-build-isolation
```

### 3. Download the Model

Download the model weights from HuggingFace:

```bash
# For the main model
huggingface-cli download stepfun-ai/stepvideo-t2v --local-dir ./models/stepvideo-t2v

# For the turbo model (faster inference)
huggingface-cli download stepfun-ai/stepvideo-t2v-turbo --local-dir ./models/stepvideo-t2v-turbo
```

## Integration Approaches

### Approach 1: Direct Pipeline Integration

Best for: Single-machine deployments, development, and testing.

```python
from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch

# Load the model
pipeline = StepVideoPipeline.from_pretrained("path/to/model").to(dtype=torch.bfloat16)
pipeline.transformer = pipeline.transformer.to("cuda")

# Setup API endpoints (for VAE and text encoding)
pipeline.setup_api(vae_url="127.0.0.1", caption_url="127.0.0.1")

# Generate a video
result = pipeline(
    prompt="A beautiful sunset over the ocean",
    num_frames=204,
    height=544,
    width=992,
    num_inference_steps=50,
    guidance_scale=9.0,
    time_shift=13.0,
)
```

**See**: `examples/simple_integration.py` for a complete example.

### Approach 2: Remote API Integration

Best for: Production deployments, microservices architecture, resource separation.

#### Step 1: Start the API Server

```bash
# On a GPU server
python api/call_remote_server.py --model_dir /path/to/model --port 8080
```

This starts two API endpoints:
- `/caption-api` - Text encoding service
- `/vae-api` - Video decoding service

#### Step 2: Use the API Client

```python
import requests
import pickle

# Encode a prompt
caption_url = "http://your-server:8080/caption-api"
data = pickle.dumps({"prompts": ["Your prompt here"]})
response = requests.get(caption_url, data=data)
embeddings = pickle.loads(response.content)
```

**See**: `examples/api_client_integration.py` for a complete example.

### Approach 3: Multi-GPU Distributed Inference

Best for: High-throughput production deployments, processing multiple videos simultaneously.

```bash
# Start API server on one GPU
python api/call_remote_server.py --model_dir /path/to/model &

# Run distributed inference on multiple GPUs
parallel=4  # Number of GPUs
url='127.0.0.1'
model_dir=/path/to/model

torchrun --nproc_per_node $parallel run_parallel.py \
    --model_dir $model_dir \
    --vae_url $url \
    --caption_url $url \
    --ulysses_degree 2 \
    --tensor_parallel_degree 2 \
    --prompt "Your prompt here" \
    --infer_steps 50 \
    --cfg_scale 9.0 \
    --time_shift 13.0
```

## Quick Start Examples

### Example 1: Simple Video Generation

```python
from stepvideo.diffusion.video_pipeline import StepVideoPipeline
import torch

# Initialize
pipeline = StepVideoPipeline.from_pretrained("./models/stepvideo-t2v")
pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.transformer = pipeline.transformer.to("cuda")
pipeline.setup_api(vae_url="127.0.0.1", caption_url="127.0.0.1")

# Generate
video = pipeline(
    prompt="A cat playing with a ball of yarn in slow motion",
    num_frames=136,
    height=544,
    width=992,
)
```

### Example 2: Custom Application Class

```python
from examples.custom_app_integration import VideoGenerationApp

# Create app instance
app = VideoGenerationApp(
    model_dir="./models/stepvideo-t2v",
    output_dir="./my_videos"
)

# Initialize
app.initialize(vae_url="127.0.0.1", caption_url="127.0.0.1")

# Generate single video
result = app.generate("A rocket launching into space")

# Batch generation
prompts = [
    "A chef preparing a meal",
    "A sunset over mountains",
    "A car racing on a track"
]
results = app.batch_generate(prompts, num_frames=136)

# Cleanup
app.cleanup()
```

**See**: `examples/custom_app_integration.py` for the full implementation.

### Example 3: Batch Processing

```python
import torch
from stepvideo.diffusion.video_pipeline import StepVideoPipeline

pipeline = StepVideoPipeline.from_pretrained("./models/stepvideo-t2v")
pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.transformer = pipeline.transformer.to("cuda")
pipeline.setup_api(vae_url="127.0.0.1", caption_url="127.0.0.1")

prompts = [
    "A beautiful landscape",
    "A futuristic city",
    "An underwater scene"
]

for i, prompt in enumerate(prompts):
    video = pipeline(
        prompt=prompt,
        output_file_name=f"video_{i:03d}",
        num_frames=204,
    )
    print(f"Generated video {i+1}/{len(prompts)}")
```

## API Reference

### StepVideoPipeline

The main pipeline class for video generation.

#### Constructor

```python
StepVideoPipeline(
    transformer: StepVideoModel,
    scheduler: FlowMatchDiscreteScheduler,
    vae_url: str = '127.0.0.1',
    caption_url: str = '127.0.0.1',
    save_path: str = './results',
    name_suffix: str = '',
)
```

#### Methods

##### `from_pretrained(model_dir: str)`

Load a pretrained model.

**Parameters:**
- `model_dir`: Path to the model directory

**Returns:** `StepVideoPipeline` instance

##### `setup_api(vae_url: str, caption_url: str)`

Configure API endpoints for VAE and text encoding.

**Parameters:**
- `vae_url`: URL of the VAE API server
- `caption_url`: URL of the caption API server

##### `__call__(...)`

Generate a video from a text prompt.

**Parameters:**
- `prompt` (str): Text description of the video to generate
- `height` (int, default=544): Video height in pixels
- `width` (int, default=992): Video width in pixels
- `num_frames` (int, default=204): Number of frames to generate
- `num_inference_steps` (int, default=50): Number of denoising steps
- `guidance_scale` (float, default=9.0): Classifier-free guidance scale
- `time_shift` (float, default=13.0): Time shift for flow matching
- `neg_magic` (str): Negative prompt modifier
- `pos_magic` (str): Positive prompt modifier
- `output_file_name` (str): Name for the output file
- `output_type` (str, default="mp4"): Output format
- `return_dict` (bool, default=True): Whether to return a dict

**Returns:** `StepVideoPipelineOutput` or tuple containing the generated video

### Remote API Endpoints

When running `api/call_remote_server.py`, two endpoints are available:

#### POST `/caption-api`

Encode text prompts into embeddings.

**Request Body (pickled):**
```python
{
    "prompts": ["text prompt 1", "text prompt 2"]
}
```

**Response (pickled):**
```python
{
    "y": tensor,              # Text embeddings
    "y_mask": tensor,         # Attention mask
    "clip_embedding": tensor  # CLIP embeddings
}
```

#### POST `/vae-api`

Decode video latents to frames.

**Request Body (pickled):**
```python
{
    "samples": tensor  # Video latents
}
```

**Response (pickled):** Decoded video tensor

## Advanced Usage

### Custom Generation Parameters

```python
# For higher quality (slower)
video = pipeline(
    prompt="Your prompt",
    num_inference_steps=100,  # More steps
    guidance_scale=12.0,      # Higher guidance
)

# For faster generation (Step-Video-T2V-Turbo)
video = pipeline(
    prompt="Your prompt",
    num_inference_steps=15,   # Fewer steps
    guidance_scale=5.0,       # Lower guidance
    time_shift=17.0,          # Turbo time shift
)

# For different resolutions
video = pipeline(
    prompt="Your prompt",
    height=768,
    width=768,
    num_frames=136,
)
```

### Memory Optimization

```python
import torch

# Use gradient checkpointing (if supported)
# pipeline.transformer.enable_gradient_checkpointing()

# Clear cache between generations
torch.cuda.empty_cache()

# Use CPU offloading for very large models
# pipeline.enable_sequential_cpu_offload()
```

### Reproducibility

```python
import torch

# Set seed for reproducibility
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Generate with fixed seed
generator = torch.Generator(device="cuda").manual_seed(seed)
video = pipeline(prompt="Your prompt", generator=generator)
```

## Production Deployment

### Recommended Architecture

For production deployments, we recommend a distributed architecture:

```
┌─────────────────┐
│  Load Balancer  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───┐
│ API  │  │ API  │  (Caption & VAE servers)
│Server│  │Server│
└──────┘  └──────┘
    │         │
    └────┬────┘
         │
    ┌────▼─────┐
    │ DiT      │  (Multi-GPU inference cluster)
    │ Workers  │
    └──────────┘
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y python3.10 python3-pip git ffmpeg

# Install Step-Video-T2V
WORKDIR /app
COPY . /app
RUN pip install -e .
RUN pip install flash-attn --no-build-isolation

# Download model (or mount as volume)
# RUN huggingface-cli download stepfun-ai/stepvideo-t2v --local-dir /models

# Start API server
CMD ["python", "api/call_remote_server.py", "--model_dir", "/models/stepvideo-t2v", "--port", "8080"]
```

Build and run:

```bash
docker build -t stepvideo-api .
docker run --gpus all -p 8080:8080 -v /path/to/models:/models stepvideo-api
```

### Performance Tuning

#### GPU Memory Requirements

| Configuration | Peak Memory | Throughput |
|--------------|-------------|------------|
| 768×768×204f | ~79 GB | ~1 video/860s |
| 544×992×204f | ~78 GB | ~1 video/743s |
| 544×992×136f | ~72 GB | ~1 video/408s |

#### Optimization Tips

1. **Use Flash Attention**: Reduces memory and improves speed
2. **Batch Sizing**: Process multiple prompts in parallel when possible
3. **Mixed Precision**: Use `torch.bfloat16` for optimal performance
4. **Model Parallelism**: Use tensor parallelism for very large models
5. **Caching**: Cache text embeddings for repeated prompts

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Solution**: Reduce resolution, frame count, or enable CPU offloading

```python
# Reduce video size
video = pipeline(prompt="...", num_frames=136, height=544, width=768)

# Or clear cache between generations
torch.cuda.empty_cache()
```

#### 2. API Connection Errors

**Solution**: Ensure API server is running and accessible

```bash
# Test API server
curl http://127.0.0.1:8080/caption-api
```

#### 3. Slow Generation

**Solution**: Use the turbo model or reduce inference steps

```python
# Use fewer steps
video = pipeline(prompt="...", num_inference_steps=30)
```

#### 4. Poor Quality Results

**Solution**: Increase inference steps and adjust guidance scale

```python
video = pipeline(
    prompt="...",
    num_inference_steps=100,
    guidance_scale=12.0,
)
```

### Getting Help

- **GitHub Issues**: [Step-Video-T2V Issues](https://github.com/stepfun-ai/Step-Video-T2V/issues)
- **Technical Report**: [ArXiv Paper](https://arxiv.org/abs/2502.10248)
- **Model Page**: [HuggingFace](https://huggingface.co/stepfun-ai/stepvideo-t2v)

## Best Practices

1. **Always validate inputs**: Check prompt length, resolution, and frame count
2. **Handle errors gracefully**: Wrap generation calls in try-except blocks
3. **Monitor resources**: Track GPU memory and processing time
4. **Use appropriate parameters**: Match parameters to your quality/speed requirements
5. **Version control**: Pin specific model versions for reproducibility
6. **Test thoroughly**: Validate integration with your specific use case

## Additional Resources

- **Example Scripts**: See the `examples/` directory for complete integration examples
- **API Server**: See `api/call_remote_server.py` for server implementation
- **Multi-GPU**: See `run_parallel.py` for distributed inference setup
- **Benchmark**: See `benchmark/` directory for evaluation tools

## License

Step-Video-T2V is released under the LICENSE file in the repository. Please review the license terms before deploying in production.
