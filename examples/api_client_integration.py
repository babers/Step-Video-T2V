"""
API Client Integration Example for Step-Video-T2V

This example demonstrates how to integrate Step-Video-T2V using the remote API
client approach for distributed inference. This is useful for production
environments where you want to separate the model serving from the application.

Usage:
    # First, start the API server on a GPU machine:
    python api/call_remote_server.py --model_dir /path/to/model --port 8080
    
    # Then, use this client to generate videos:
    python examples/api_client_integration.py --api_url http://your-server-ip --prompt "Your prompt here"
"""

import requests
import pickle
import argparse
import numpy as np
from typing import Optional, Dict, Any


class StepVideoAPIClient:
    """
    Client for interacting with Step-Video-T2V remote API servers.
    
    This client handles communication with both the caption API (text encoding)
    and VAE API (video decoding) servers.
    """
    
    def __init__(self, api_url: str, port: int = 8080):
        """
        Initialize the API client.
        
        Args:
            api_url: Base URL of the API server (e.g., "http://127.0.0.1")
            port: Port number of the API server (default: 8080)
        """
        self.api_url = api_url.rstrip('/')
        self.port = port
        self.caption_endpoint = f"{self.api_url}:{port}/caption-api"
        self.vae_endpoint = f"{self.api_url}:{port}/vae-api"
    
    def encode_prompt(self, prompts: list) -> Dict[str, Any]:
        """
        Encode text prompts using the caption API.
        
        Args:
            prompts: List of text prompts to encode
            
        Returns:
            Dictionary containing encoded embeddings and attention masks
        """
        data = {"prompts": prompts}
        data_bytes = pickle.dumps(data)
        
        response = requests.get(self.caption_endpoint, data=data_bytes)
        if response.status_code != 200:
            raise Exception(f"Caption API error: {response.status_code}")
        
        result = pickle.loads(response.content)
        return result
    
    def decode_latents(self, latents: np.ndarray) -> np.ndarray:
        """
        Decode video latents using the VAE API.
        
        Args:
            latents: Video latents tensor to decode
            
        Returns:
            Decoded video frames
        """
        data = {"samples": latents}
        data_bytes = pickle.dumps(data)
        
        response = requests.get(self.vae_endpoint, data=data_bytes)
        if response.status_code != 200:
            raise Exception(f"VAE API error: {response.status_code}")
        
        result = pickle.loads(response.content)
        return result


def generate_video_with_api(
    api_url: str,
    prompt: str,
    port: int = 8080,
    num_frames: int = 204,
    height: int = 544,
    width: int = 992,
):
    """
    Generate a video using the Step-Video-T2V API.
    
    Args:
        api_url: Base URL of the API server
        prompt: Text prompt for video generation
        port: Port number of the API server
        num_frames: Number of frames to generate
        height: Video height in pixels
        width: Video width in pixels
        
    Returns:
        Generated video frames
    """
    # Initialize API client
    client = StepVideoAPIClient(api_url, port)
    
    # Encode the prompt
    print(f"Encoding prompt: '{prompt}'")
    embeddings = client.encode_prompt([prompt])
    
    print("Prompt encoded successfully!")
    print(f"Embedding shape: {embeddings['y'].shape}")
    
    # Note: The actual denoising loop would be run on the client side
    # using the transformer model, which requires having the model locally.
    # For a pure API-based approach, you would need to extend the API server
    # to include a full generation endpoint.
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Step-Video-T2V API Client Integration Example")
    parser.add_argument("--api_url", type=str, default="http://127.0.0.1", help="API server URL")
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--num_frames", type=int, default=204, help="Number of frames")
    parser.add_argument("--height", type=int, default=544, help="Video height")
    parser.add_argument("--width", type=int, default=992, help="Video width")
    
    args = parser.parse_args()
    
    # Test the API connection
    print(f"Connecting to Step-Video-T2V API at {args.api_url}:{args.port}")
    
    embeddings = generate_video_with_api(
        api_url=args.api_url,
        prompt=args.prompt,
        port=args.port,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    
    print("\nAPI integration test completed successfully!")
    print("\nNote: For full video generation, you need to run the denoising loop")
    print("on the client side with the transformer model, or extend the API")
    print("server to include a complete generation endpoint.")


if __name__ == "__main__":
    main()
