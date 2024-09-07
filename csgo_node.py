# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,

)
from transformers import BlipProcessor, BlipForConditionalGeneration

from .CSGO.ip_adapter.utils import BLOCKS as BLOCKS
from .CSGO.ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
from .CSGO.ip_adapter.utils import resize_content
from .CSGO.ip_adapter import CSGO


import folder_paths
from comfy.utils import common_upscale
from comfy.clip_vision import  load

MAX_SEED = np.iinfo(np.int32).max


node_cur_path = os.path.dirname(os.path.abspath(__file__))
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
weight_dtype = torch.float16

def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def pil2narry(img):
    narry = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return narry

def tensor_upscale2pil(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor2pil(samples)
    return img_pil

def tensor_upscale(img_tensor, width, height): #torch tensor
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return samples

class Blip_Loader:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "blip_repo": ("STRING", {"default": "Salesforce/blip-image-captioning-large"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "MODEL",)
    RETURN_NAMES = ("blip_processor", "blip_model",)
    FUNCTION = "test"
    CATEGORY = "CSGO_Wrapper"

    def test(self, blip_repo,):
        blip_processor = BlipProcessor.from_pretrained(blip_repo)
        blip_model = BlipForConditionalGeneration.from_pretrained(blip_repo).to(device)
        return (blip_processor,blip_model,)
        
    
class CSGO_Loader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_cpkt":(["none"]+folder_paths.get_filename_list("checkpoints"),),
                "clip_vision":(["none"]+folder_paths.get_filename_list("clip_vision"),),
                "vae_id":(["none"]+folder_paths.get_filename_list("vae"),),
                "controlnet_repo": ("STRING", {"default": "TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic" }),
                "csgo_ckpt": (["none"]+folder_paths.get_filename_list("checkpoints"),),
                "num_content_tokens": ("INT", {
                    "default": 4,
                    "min": 1,  # Minimum value
                    "max": 512,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "num_style_tokens": ("INT", {
                    "default": 32,
                    "min": 1,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("csgo",)
    FUNCTION = "test"
    CATEGORY = "CSGO_Wrapper"

    def test(self,base_cpkt,clip_vision,vae_id,controlnet_repo,csgo_ckpt,num_content_tokens,num_style_tokens):
        if csgo_ckpt=="none" or base_cpkt=="none" :
            raise "need weight"
            
        csgo_ckpt=folder_paths.get_full_path("checkpoints",csgo_ckpt) #获取绝对路径
        ckpt_path = folder_paths.get_full_path("checkpoints", base_cpkt)  # 获取绝对路径
        clip_vision=folder_paths.get_full_path("clip_vision", clip_vision)  # 获取绝对路径
        img_encoder=load(clip_vision)
        model_config = os.path.join(node_cur_path, "local_repo")
        original_config_file = os.path.join(node_cur_path, 'configs', 'sd_xl_base.yaml')
        controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=torch.float16, use_safetensors=True)
        vae_id = folder_paths.get_full_path("vae", vae_id)
        vae_config = os.path.join(node_cur_path, "local_repo","vae")
        vae = AutoencoderKL.from_single_file(vae_id, config=vae_config,torch_dtype=torch.float16)
        try:
            pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                ckpt_path,config=model_config, original_config=original_config_file,vae=vae, controlnet=controlnet,
                torch_dtype=torch.float16)
        except:
            try:
                pipe = StableDiffusionXLControlNetPipeline.from_single_file(
                    ckpt_path,config=model_config, original_config_file=original_config_file,vae=vae, controlnet=controlnet,
                    torch_dtype=torch.float16)
            except:
                raise "load pipe error!,check you diffusers"
        
        pipe.enable_vae_tiling()
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        if device != "mps":
            pipe.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        target_content_blocks = BLOCKS['content']
        target_style_blocks = BLOCKS['style']
        controlnet_target_content_blocks = controlnet_BLOCKS['content']
        controlnet_target_style_blocks = controlnet_BLOCKS['style']
        config_path = os.path.join(node_cur_path, "configs", "config.json")
        image_encoder_config = OmegaConf.load(config_path)
       
        csgo = CSGO(pipe, img_encoder, csgo_ckpt, device, image_encoder_config,num_content_tokens=num_content_tokens, num_style_tokens=num_style_tokens,
                    target_content_blocks=target_content_blocks, target_style_blocks=target_style_blocks,
                    controlnet_adapter=True,
                    controlnet_target_content_blocks=controlnet_target_content_blocks,
                    controlnet_target_style_blocks=controlnet_target_style_blocks,
                    content_model_resampler=True,
                    style_model_resampler=True,
                    )
        return (csgo,)
    
class CSGO_Sampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "content_image": ("IMAGE",),
                "style_image": ("IMAGE",),
                "csgo": ("MODEL",),
                "prompt": ("STRING", {"multiline": True,"default": "a cat"}),
                "negative_prompt": ("STRING", {"multiline": True,"default": "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry"}),
                "num_sample": ("INT", {
                    "default": 1,
                    "min": 1,  # Minimum value
                    "max": 100,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,  # Minimum value
                    "max": 4096,  # Maximum value
                    "step": 64,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "content_scale": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number",
                }),
                "style_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number",
                }),
                "cfg": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 30.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number",
                }),
                "steps": ("INT", {
                    "default": 25,
                    "min": 1,  # Minimum value
                    "max": 100,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": -1,  # Minimum value
                    "max": MAX_SEED,  # Maximum value
                }),
                "controlnet_scale": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    # The value representing the precision to round to, will be set to the step value by default. Can be set to False to disable rounding.
                    "display": "number",
                }),
                "text_only": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "blip_processor": ("MODEL",),
                "blip_model": ("MODEL",),
                         },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "test"
    CATEGORY = "CSGO_Wrapper"
    
    def test(self,content_image,style_image, csgo, prompt,negative_prompt,num_sample, width,height,content_scale,style_scale,cfg,steps,seed,controlnet_scale,text_only,**kwargs):
        
        blip_processor=kwargs.get("blip_processor",None)
        blip_model=kwargs.get("blip_model",None)
        
        #向量转图片
        #style_image= tensor2pil(style_image)
        #content_image= tensor2pil(content_image)
        content_image_pil = tensor_upscale2pil(content_image, width, height)
        style_image =tensor_upscale(style_image, width, height)  # torch tensor

        content_image=tensor_upscale(content_image, width, height)  # torch tensor
        
        
        
        #clip vison
        if blip_processor and blip_model:  #判断模型是否存在，不存在则跑文本驱动流程
            with torch.no_grad():
                inputs = blip_processor(content_image, return_tensors="pt").to(device)
                out = blip_model.generate(**inputs)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)
                del blip_model,blip_processor
                torch.cuda.empty_cache()
                
        else:
            caption = prompt
            if text_only:
                content_image_pil = Image.fromarray(
                    np.zeros((content_image_pil.size[0], content_image_pil.size[1], 3), dtype=np.uint8)).convert('RGB')
                content_image=pil2narry(content_image_pil)
                
                
        
        #width, height, content_image = resize_content(content_image)#?
        
        images = csgo.generate(pil_content_image=content_image, pil_style_image=style_image,
                               prompt=caption,
                               negative_prompt=negative_prompt,
                               height=height,
                               width=width,
                               content_scale=content_scale,
                               style_scale=style_scale,
                               guidance_scale=cfg,
                               num_images_per_prompt=num_sample,
                               num_samples=num_sample,
                               num_inference_steps=steps,
                               seed=seed,
                               image= content_image_pil,
                               controlnet_conditioning_scale=controlnet_scale,
                               )[0]
        images=pil2narry(images)
        return (images,)
    #@classmethod
    #def IS_CHANGED(s, image, string_field, int_field, float_field, print_to_screen):
    #    return ""

# Set the web directory, any .js file in that directory will be loaded by the frontend as a frontend extension
# WEB_DIRECTORY = "./somejs"

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Blip_Loader":Blip_Loader,
    "CSGO_Loader": CSGO_Loader,
    "CSGO_Sampler":CSGO_Sampler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Blip_Loader":"Blip_Loader",
    "CSGO_Loader": "CSGO_Loader",
    "CSGO_Sampler":"CSGO_Sampler"
}
