import requests
import base64
import io
from PIL import Image

class AuroraImagination:
    def __init__(self, api_url="http://127.0.0.1:7860"):
        self.url = f"{api_url}/sdapi/v1/txt2img"
        # This is the "Visual DNA" we will refine
        self.base_persona = (
            "Aurora, a sleek woman with teal eyes, sharp bob haircut, "
            "wearing futuristic tech-wear, cinematic lighting, masterpiece, 8k"
        )

    def imagine(self, prompt, save_path="static/gen/latest.png"):
        payload = {
            "prompt": f"{self.base_persona}, {prompt}",
            "negative_prompt": "easynegative, deformed, blurry, extra fingers",
            "steps": 25,
            "cfg_scale": 7,
            "width": 512,
            "height": 768,
            "restore_faces": True
        }
        
        response = requests.post(self.url, json=payload)
        r = response.json()
        
        for i in r['images']:
            image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
            image.save(save_path)
            
        return save_path