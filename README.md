# Pixel Prompt Backend

This is the backend component of Pixel Prompt, a versatile application designed to handle a wide range of ML workloads. Built using FastAPI and Docker, this backend provides a robust and scalable foundation for hosting and managing ML models and associated APIs.

It uses the StableDiffusionPipeline from huggingface to generate images using diffusion and works in conjunction with the frontend. 

## Installation :hammer:

Install pytorch separately from their [main page](https://pytorch.org/) as the pip repository is out-of-sync at times.

```shell
git clone https://github.com/hatmanstack/pixel-prompt-frontend.git
cd pixel-prompt-frontend
python -m venv venv

**Linux**
source venv/bin/activate

**Windows**
.\venv\scripts\activate

pip install -r requirements.txt
python main.py
```

After running `main.py` a server should be running at port 8085 locally.  The port the frontend needs to call with `http://localhost:8085/api`. 

Add your HF_TOKEN in a .env file in the app folder.

## Models :sparkles:

All the models are opensource and available on HuggingFace.

### Diffusion

- **stabilityai/stable-diffusion-xl-base-1.0**
- **stabilityai/stable-diffusion-xl-refiner-1.0**
- **prompthero/openjourney**
- **dreamlike-art/dreamlike-photoreal-2.0**
- **nitrosocke/Arcane-Diffusion**
- **dallinmackay/Van-Gogh-diffusion**
- **nousr/robo-diffusion**

### Prompts

- **Gustavosta/MagicPrompt-Stable-Diffusion**

## License

This project is licensed under the [MIT License](LICENSE)

## Acknowledgments :trophy:

- This application is built with Diffusers, an awesome library provided by [HuggingFace](https://huggingface.co)

<p align="center"><img src="https://github.com/HatmanStack/pixel-prompt-backend/blob/main/logo.png" alt="Image 4"></p>