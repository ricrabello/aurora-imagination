import torch
from diffusers import DiffusionPipeline
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from io import BytesIO
import base64

app = FastAPI()

# Load Juggernaut XL once at startup
model_id = "RunDiffusion/Juggernaut-XL-v9"
hf_token = os.environ.get("HF_TOKEN")

print(f"🚀 Loading {model_id} onto your 5070 Ti...")
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_auth_token=hf_token
)
pipe.to("cuda")

class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = None

@app.post("/sdapi/v1/txt2img")
async def generate_image(request: ImageRequest):
    try:
        # Standard high-end negative prompt
        default_neg = "lowres, bad anatomy, bad hands, worst quality, low quality, blurry, plastic, cgi"
        neg = request.negative_prompt if request.negative_prompt else default_neg

        image = pipe(
            prompt=request.prompt,
            negative_prompt=neg,
            num_inference_steps=35,
            guidance_scale=7.0,
            width=1024,
            height=1024
        ).images[0]

        # Convert to Base64 for the API response
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return {"images": [img_str]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)