import torch
import fastapi
import base64
import os
from dotenv import load_dotenv
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from io import BytesIO

load_dotenv()

token = os.getenv("HF_TOKEN")

app = FastAPI()

class Item(BaseModel):
    prompt: str
    steps: int
    guidance: float
    modelID: str

@app.post("/api")
async def inference(item: Item):
    print(item.prompt)
    if "stable-diffusion" in item.modelID:
        prompt = item.prompt
    if "dallinmackay" in item.modelID:
        prompt = "lvngvncnt, " + item.prompt
    if "nousr" in item.modelID:
        prompt = "nousr robot, " + item.prompt
    if "nitrosocke" in item.modelID:
        prompt = "arcane, " + item.prompt
    if "dreamlike" in item.modelID:
        prompt = "photo, " + item.prompt
    if "prompthero" in item.modelID:
        prompt = "mdjrny-v4 style, " + item.prompt 

    negative_prompt = "blurry, low quality, ugly, deformed, malformed, lowres, mutant, mutated, disfigured, compressed, noise, artifacts, dithering, simple, watermark, text, font, signage, collage, pixel"
    data = {"inputs":prompt, "negative_prompt": negative_prompt, "options":{"wait_for_model": True, "use_cache": False}}
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(item.modelID,safety_checker=None,
            torch_dtype=torch.float16,
            use_safetensors=True).to("cuda")
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = StableDiffusionPipeline.from_pretrained(item.modelID, safety_checker=None)
    
    image = pipe(prompt=prompt, num_inference_steps=item.steps, guidance=item.guidance).images[0]

    image.save("response.png")
    with open('response.png', 'rb') as f:
        base64image = base64.b64encode(f.read())
    
    return {"output": base64image}


