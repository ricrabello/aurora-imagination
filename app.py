
from diffusers import StableDiffusionPipeline
import torch
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion Image Generator")
    parser.add_argument('--prompt', type=str, required=True, help='Positive prompt for image generation')
    parser.add_argument('--negative', type=str, default=None, help='Negative prompt (optional)')
    parser.add_argument('--output', type=str, default="output.png", help='Output filename (default: output.png)')
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("WARNING: HF_TOKEN environment variable not set. Model download might fail for private/gated models.")
        print("Please create a token at https://huggingface.co/settings/tokens and set the HF_TOKEN environment variable.")

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        use_auth_token=hf_token
    )
    # Disable safety checker (NSFW filter)
    if hasattr(pipe, 'safety_checker'):
        pipe.safety_checker = lambda images, **kwargs: (images, [False]*len(images))
    try:
        pipe.to("cuda")
    except RuntimeError:
        print("CUDA not available, falling back to CPU. This may be slow.")
        pipe.to("cpu")

    # Generate image
    if args.negative:
        image = pipe(args.prompt, negative_prompt=args.negative).images[0]
    else:
        image = pipe(args.prompt).images[0]

    image.save(args.output)
    print(f"Image saved as {args.output}")

if __name__ == "__main__":
    main()