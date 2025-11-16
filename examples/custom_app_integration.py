"""
Custom Application Integration Example

This example shows how to build a custom video generation application class
that encapsulates Step-Video-T2V functionality, making it easy to integrate
into larger applications, web services, or batch processing pipelines.

Usage:
    python examples/custom_app_integration.py --model_dir /path/to/model
"""

import torch
import argparse
import os
from typing import Optional, List, Dict, Any
from pathlib import Path
from stepvideo.diffusion.video_pipeline import StepVideoPipeline


class VideoGenerationApp:
    """
    Custom video generation application wrapping Step-Video-T2V.
    
    This class provides a clean interface for integrating Step-Video-T2V
    into your custom application with features like:
    - Batch processing
    - Configuration management
    - Error handling
    - Resource management
    """
    
    def __init__(
        self,
        model_dir: str,
        output_dir: str = "./generated_videos",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize the video generation application.
        
        Args:
            model_dir: Path to the Step-Video-T2V model directory
            output_dir: Directory to save generated videos
            device: Device to run inference on (default: auto-detect)
            dtype: Data type for model weights (default: bfloat16)
        """
        self.model_dir = model_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.dtype = dtype
        self.pipeline = None
        
        # Default generation parameters
        self.default_params = {
            "num_frames": 204,
            "height": 544,
            "width": 992,
            "num_inference_steps": 50,
            "guidance_scale": 9.0,
            "time_shift": 13.0,
        }
        
    def initialize(self, vae_url: str = "127.0.0.1", caption_url: str = "127.0.0.1"):
        """
        Initialize the model pipeline.
        
        Args:
            vae_url: URL of the VAE API server
            caption_url: URL of the caption API server
        """
        print(f"Loading Step-Video-T2V model from {self.model_dir}...")
        self.pipeline = StepVideoPipeline.from_pretrained(self.model_dir).to(dtype=self.dtype)
        self.pipeline.transformer = self.pipeline.transformer.to(self.device)
        self.pipeline.setup_api(vae_url=vae_url, caption_url=caption_url)
        self.pipeline.video_processor.save_path = str(self.output_dir)
        print("Model initialized successfully!")
        
    def generate(
        self,
        prompt: str,
        output_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a video from a text prompt.
        
        Args:
            prompt: Text description of the video to generate
            output_name: Custom name for the output file (optional)
            **kwargs: Override default generation parameters
            
        Returns:
            Dictionary containing generation results and metadata
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        # Merge default parameters with user-provided overrides
        params = {**self.default_params, **kwargs}
        
        # Use prompt as filename if not specified
        if output_name is None:
            output_name = prompt[:50]
        
        print(f"\nGenerating video: '{prompt}'")
        print(f"Parameters: {params}")
        
        try:
            result = self.pipeline(
                prompt=prompt,
                output_file_name=output_name,
                **params
            )
            
            return {
                "success": True,
                "prompt": prompt,
                "output_path": str(self.output_dir / f"{output_name}.mp4"),
                "parameters": params,
                "video": result.video,
            }
            
        except Exception as e:
            print(f"Error generating video: {e}")
            return {
                "success": False,
                "prompt": prompt,
                "error": str(e),
            }
    
    def batch_generate(
        self,
        prompts: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple videos from a list of prompts.
        
        Args:
            prompts: List of text descriptions
            **kwargs: Generation parameters to apply to all prompts
            
        Returns:
            List of generation results for each prompt
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"Processing {i+1}/{len(prompts)}")
            print(f"{'='*60}")
            
            result = self.generate(prompt, **kwargs)
            results.append(result)
            
        return results
    
    def set_default_params(self, **kwargs):
        """
        Update default generation parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        self.default_params.update(kwargs)
        print(f"Updated default parameters: {self.default_params}")
    
    def cleanup(self):
        """
        Clean up resources and free GPU memory.
        """
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Resources cleaned up.")


def example_single_generation():
    """Example: Generate a single video"""
    print("\n" + "="*60)
    print("Example 1: Single Video Generation")
    print("="*60)
    
    app = VideoGenerationApp(
        model_dir="path/to/model",
        output_dir="./my_videos"
    )
    
    # Initialize the model
    app.initialize(vae_url="127.0.0.1", caption_url="127.0.0.1")
    
    # Generate a video
    result = app.generate(
        prompt="A beautiful sunset over the ocean with waves crashing",
        num_frames=136,  # Override default
        guidance_scale=7.5,  # Override default
    )
    
    if result["success"]:
        print(f"\nVideo saved to: {result['output_path']}")
    
    # Clean up
    app.cleanup()


def example_batch_generation():
    """Example: Generate multiple videos"""
    print("\n" + "="*60)
    print("Example 2: Batch Video Generation")
    print("="*60)
    
    app = VideoGenerationApp(model_dir="path/to/model")
    app.initialize()
    
    # Set custom defaults for this batch
    app.set_default_params(num_frames=136, num_inference_steps=30)
    
    prompts = [
        "A cat playing with a ball of yarn",
        "A rocket launching into space",
        "A chef preparing a delicious meal",
    ]
    
    results = app.batch_generate(prompts)
    
    # Print summary
    successful = sum(1 for r in results if r["success"])
    print(f"\nGenerated {successful}/{len(results)} videos successfully")
    
    app.cleanup()


def example_custom_workflow():
    """Example: Custom workflow with error handling"""
    print("\n" + "="*60)
    print("Example 3: Custom Workflow")
    print("="*60)
    
    app = VideoGenerationApp(
        model_dir="path/to/model",
        output_dir="./campaign_videos"
    )
    
    try:
        app.initialize()
        
        # Your custom workflow
        prompts_config = [
            {"prompt": "Product showcase in modern setting", "num_frames": 204},
            {"prompt": "Customer testimonial scene", "num_frames": 136},
            {"prompt": "Brand logo animation", "num_frames": 68},
        ]
        
        for config in prompts_config:
            result = app.generate(**config)
            
            if result["success"]:
                # Do something with the generated video
                # e.g., post-processing, uploading to cloud storage, etc.
                print(f"✓ Generated: {result['output_path']}")
            else:
                print(f"✗ Failed: {result['error']}")
                
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        app.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Custom Video Generation Application")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory")
    parser.add_argument("--output_dir", type=str, default="./generated_videos", help="Output directory")
    parser.add_argument("--vae_url", type=str, default="127.0.0.1", help="VAE API URL")
    parser.add_argument("--caption_url", type=str, default="127.0.0.1", help="Caption API URL")
    parser.add_argument("--prompt", type=str, help="Text prompt for video generation")
    parser.add_argument("--prompts_file", type=str, help="File containing prompts (one per line)")
    
    args = parser.parse_args()
    
    # Create and initialize the app
    app = VideoGenerationApp(
        model_dir=args.model_dir,
        output_dir=args.output_dir
    )
    app.initialize(vae_url=args.vae_url, caption_url=args.caption_url)
    
    try:
        if args.prompt:
            # Single video generation
            result = app.generate(args.prompt)
            if result["success"]:
                print(f"\n✓ Video saved to: {result['output_path']}")
            else:
                print(f"\n✗ Generation failed: {result['error']}")
                
        elif args.prompts_file:
            # Batch generation from file
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            
            results = app.batch_generate(prompts)
            successful = sum(1 for r in results if r["success"])
            print(f"\n✓ Generated {successful}/{len(results)} videos successfully")
            
        else:
            print("Please provide either --prompt or --prompts_file")
            print("\nTo see integration examples, check the example_* functions in this file.")
            
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()
