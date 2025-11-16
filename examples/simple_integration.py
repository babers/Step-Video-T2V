"""
Simple Integration Example for Step-Video-T2V

This example demonstrates how to integrate Step-Video-T2V into your custom
video generation application using the pipeline directly.

Usage:
    python examples/simple_integration.py --model_dir /path/to/model --prompt "Your prompt here"
"""

import torch
import argparse
from stepvideo.diffusion.video_pipeline import StepVideoPipeline


def generate_video(
    model_dir: str,
    prompt: str,
    output_path: str = "./output",
    num_frames: int = 204,
    height: int = 544,
    width: int = 992,
    num_inference_steps: int = 50,
    guidance_scale: float = 9.0,
    time_shift: float = 13.0,
    seed: int = 1234,
):
    """
    Generate a video using Step-Video-T2V model.
    
    Args:
        model_dir: Path to the downloaded model directory
        prompt: Text prompt for video generation
        output_path: Directory to save the generated video
        num_frames: Number of frames to generate (default: 204)
        height: Video height in pixels (default: 544)
        width: Video width in pixels (default: 992)
        num_inference_steps: Number of denoising steps (default: 50)
        guidance_scale: Classifier-free guidance scale (default: 9.0)
        time_shift: Time shift parameter for flow matching (default: 13.0)
        seed: Random seed for reproducibility
        
    Returns:
        Generated video tensor or path to saved video file
    """
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Load the pipeline
    print(f"Loading Step-Video-T2V model from {model_dir}...")
    pipeline = StepVideoPipeline.from_pretrained(model_dir).to(dtype=torch.bfloat16)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.transformer = pipeline.transformer.to(device)
    
    # Note: For production use, you should set up the API servers for VAE and caption
    # See api/call_remote_server.py for details
    # For this example, we assume API servers are running on localhost
    pipeline.setup_api(vae_url="127.0.0.1", caption_url="127.0.0.1")
    
    # Set output path
    pipeline.video_processor.save_path = output_path
    
    print(f"Generating video with prompt: '{prompt}'")
    print(f"Parameters: {num_frames} frames, {height}x{width}, {num_inference_steps} steps")
    
    # Generate the video
    result = pipeline(
        prompt=prompt,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        time_shift=time_shift,
        output_file_name=prompt[:50],  # Use first 50 chars of prompt as filename
    )
    
    print(f"Video generation complete! Saved to {output_path}")
    return result.video


def main():
    parser = argparse.ArgumentParser(description="Simple Step-Video-T2V Integration Example")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--output_path", type=str, default="./output", help="Output directory")
    parser.add_argument("--num_frames", type=int, default=204, help="Number of frames")
    parser.add_argument("--height", type=int, default=544, help="Video height")
    parser.add_argument("--width", type=int, default=992, help="Video width")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--cfg_scale", type=float, default=9.0, help="CFG scale")
    parser.add_argument("--time_shift", type=float, default=13.0, help="Time shift")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    
    args = parser.parse_args()
    
    video = generate_video(
        model_dir=args.model_dir,
        prompt=args.prompt,
        output_path=args.output_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg_scale,
        time_shift=args.time_shift,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
