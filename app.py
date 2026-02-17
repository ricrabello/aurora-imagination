import torch
from diffusers import DiffusionPipeline, AutoencoderKL
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="SDXL Photorealistic Generator")
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--negative', type=str, default=None)
    parser.add_argument('--output', type=str, default="output.png")
    args = parser.parse_args()

    hf_token = os.environ.get("HF_TOKEN")

    # Load Juggernaut XL - The gold standard for photorealism
    model_id = "RunDiffusion/Juggernaut-XL-v9"
    
    print(f"🚀 Loading {model_id} onto your 5070 Ti...")
    
    # We use DiffusionPipeline as the universal loader
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_auth_token=hf_token
    )
    
    # Optimization for Blackwell Architecture (RTX 50-series)
    pipe.to("cuda")
    
    # High-end Negative Prompt
    default_negative = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "jpeg artifacts, signature, watermark, blurry, plastic, cgi, render, "
        "3d, illustration, cartoon, doll, fake skin, deformed, disfigured, out of frame, "
        "ugly, tiling, poorly drawn hands, poorly drawn face, mutation, mutated, "
        "extra limbs, extra arms, extra legs, fused fingers, too many fingers, "
        "long neck, long body, low contrast, bad lighting, overexposed, underexposed"
    )
    neg_prompt = args.negative if args.negative else default_negative

    print("📸 Generating 1024x1024 professional photography...")
    
    # SDXL handles 'Full Body' much better than v1.5
    image = pipe(
        prompt=args.prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=35, # SDXL likes 30-40 steps
        guidance_scale=7.0,
        width=1024,
        height=1024
    ).images[0]

    image.save(args.output)
    print(f"✅ Success! Masterpiece saved as {args.output}")

if __name__ == "__main__":
    main()