# ComfyUI_CSGO_Wrapper
You can using InstantX's CSGO in comfyUI

**CSGO From: [link](https://github.com/instantX-research/CSGO)**

Update
---

**2024/09/07**
* fix runway loader error, using single clip_vision /controlnet weight now；
* 修复runway跑路导致的diffuser加载报错，现在直接使用IP adapter的SDXL 图片解码单体模型（请放在clip_vision目录下）和单体controlnet SDLX模型（请放在controlnet目录下）；
* vae is not necessary,if you get black img ,please using vae  
* vae并不是必须的，但是如果出现黑图或者解码不正确，才使用；  

1.Installation
-----
  In the ./ComfyUI /custom_node directory, run the following:   
```
git clone https://github.com/smthemex/ComfyUI_CSGO_Wrapper.git
```  
  
2.requirements  
----
For ComfyUI users, all libraries in the requirements file should be available. If not, please uncomment the # and reinstall
```
pip install -r requirements.txt
```
3 Need  model 
----
3.1 base SDXl ckpt  and vae and clip_vision and controlnet        
 h94/IP-Adapter/sdxl_models [link](https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models/image_encoder)  
 TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic [link](https://huggingface.co/TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic)
no need config   
```
├── ComfyUI/models/checkpoints/
|      ├──any SDXL weights  # for example: Jumpernaut XL_v9-RunDiffusionPhoto_v2.safetensors
├── ComfyUI/models/vae/
|      ├──any SDXL vae weights  # sdxl.vae.safetensors 
├── ComfyUI/models/clip_vision/
|      ├──model.safetensors  # h94/IP-Adapter/sdxl_models/model.safetensors
├── ComfyUI/models/controlnet /
|      ├──TTPLANET_Controlnet_Tile_realistic_v2_fp16.safetensors  # TTPlanet/TTPLanet_SDXL_Controlnet_Tile_Realistic
```
3.2 main ckpt      
CSGO models [link](https://huggingface.co/InstantX/CSGO/tree/main)
```
├── ComfyUI/models/checkpoints/
|      ├──acsgo_4_32.bin  #need token 4/32
|      ├──acsgo.bin   #need token 4/16
```

3.3 if using LLM  (unnecessary)   
Salesforce/blip-image-captioning-large  [link](https://huggingface.co/Salesforce/blip-image-captioning-large/tree/main)
```
├── any path dir/
|             ├── model.safetensors
|             ├── config.json
|             ├── preprocessor_config.json
|             ├── special_tokens_map.json
|             ├── tokenizer.json
|             ├── tokenizer_config.json
|             ├── vocab.txt
```
or change prompt input using any other LLM.

4 Example
----
conternt+style img   
![](https://github.com/smthemex/ComfyUI_CSGO_Wrapper/blob/main/example/content_style_img.png)  
style img  + prompt   
![](https://github.com/smthemex/ComfyUI_CSGO_Wrapper/blob/main/example/txt_only.png)
conternt+style img  +llm   new  最新  
![](https://github.com/smthemex/ComfyUI_CSGO_Wrapper/blob/main/example/new.png)


5 Function Description of Nodes  
---
* content_tokens:  keep in 4;   
* style_tokens:   if using csgo.bin keep num "16",if using csgo_4_32.bin keep num  "32" ;  
* text_only: if not using content img ;    
* Blip is not necessary;  


6 Citation
------
CSGO
``` python  
@article{xing2024csgo,
       title={CSGO: Content-Style Composition in Text-to-Image Generation}, 
       author={Peng Xing and Haofan Wang and Yanpeng Sun and Qixun Wang and Xu Bai and Hao Ai and Renyuan Huang and Zechao Li},
       year={2024},
       journal = {arXiv 2408.16766},
}
}
```

