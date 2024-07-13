# Pixel Prompt Backend

This is the backend component of Pixel Prompt, a versatile application designed to handle a wide range of ML workloads. Built using FastAPI and Docker, this backend provides a robust and scalable foundation for hosting and managing ML models and associated APIs.

It uses the StableDiffusionPipeline from huggingface to generate images using diffusion and works in conjunction with the frontend. 
A blog post talking about the architecture can be found here: [Cloud Bound](https://medium.com/@HatmanStack/cloud-bound-react-native-and-fastapi-ml-684a658f967a).  

## Preview :zap:

To preview the application visit the hosted version on the Hugging Face Spaces platform [here](https://hatman-pixel-prompt.hf.space).

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

- **stabilityai/stable-diffusion-3-medium**
- **stabilityai/stable-diffusion-xl-base-1.0**
- **fluently/Fluently-XL-Final**
- **nerijs/pixel-art-xl**
- **Fictiverse/Voxel_XL_Lora**
- **dallinmackay/Van-Gogh-diffusion**
- **gsdf/Counterfeit-V2.5**
- **digiplay/AbsoluteReality_v1.8.1**
- **dreamlike-art/dreamlike-photoreal-2.0**
- **digiplay/Acorn_Photo_v1**

### Prompts

- **mistralai/Mistral-7B-Instruct-v0.3**
- **Gustavosta/MagicPrompt-Stable-Diffusion**
- **meta-llama/Meta-Llama-3-70B-Instruct**

## License

This project is licensed under the [MIT License](LICENSE)

## Acknowledgments :trophy:

<p align="center">This application is built with Diffusers, an awesome library provided by <a href="https://huggingface.co">HuggingFace</a> </br><img src="https://github.com/HatmanStack/pixel-prompt-backend/blob/main/logo.png" alt="Image 4"></p>