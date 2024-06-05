from fastapi import FastAPI, File, UploadFile
import torch
import fastapi
import base64
import requests
import os
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel
from diffusers import StableDiffusionPipeline
from io import BytesIO
from typing import Optional

load_dotenv()

token = os.getenv("HF_TOKEN")

app = FastAPI()

class Item(BaseModel):
    prompt: str
    steps: int
    guidance: float
    modelID: str

class promptType(BaseModel):
    prompt: str
    modelID: str

@app.post("/inferencePrompt")
async def inferencePrompt(item: promptType):
    input = item.prompt
    modelID = item.modelID
    API_URL = f'https://api-inference.huggingface.co/models/{modelID}'
    headers = {"Authorization": f"Bearer {token}"}    
    parameters = {"return_full_text":False,"max_new_tokens":300}
    options = {"use_cache": False, "wait_for_model": True}
    response = requests.post(API_URL, headers=headers, \
        json={"inputs":input, "parameters": parameters,"options": options})
        
    if response.status_code != 200:
        print(response.json().get("error_type"), response.status_code)
        return {"error": response.json().get("error")}
    
    return response.json()

@app.post("/api")
async def inference(item: Item, image: Optional[UploadFile] = File(None)):
    contents = None
    if image:
        contents = await image.read()
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

    negative_prompt = "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"
    data = {"inputs": prompt, "negative_prompt": negative_prompt, "options": {"wait_for_model": True, "use_cache": False}}
    if 'holder' not in item.scale.keys():
        data["set_ip_adapter_scale"] = item.scale
        data["ip_adapter_image"] = contents
    
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
