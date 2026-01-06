import os
import json
import numpy as np
import torch
import hashlib
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import nodes
import comfy.model_management
import comfy.samplers

# --- CACHE DEGLI HASH ---
HASH_CACHE = {}

def get_sha256_hash(file_path):
    if not file_path or not os.path.exists(file_path):
        return "Unknown"
    if file_path in HASH_CACHE:
        return HASH_CACHE[file_path]
    try:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        short_hash = hasher.hexdigest()[:10]
        HASH_CACHE[file_path] = short_hash
        return short_hash
    except:
        return "Unknown"

def clean_name(path):
    if not path or path == "None": return None
    return os.path.splitext(os.path.basename(path))[0]

# ==========================================
# LOGICA DI SALVATAGGIO ECO-SISTEMA FURREY PRO
# ==========================================
def save_furrey_logic(images, filename_prefix, prompt_data, extra_pnginfo, pos_text, neg_text, steps, sampler, scheduler, cfg, seed, model_name, denoise, mode):
    output_dir = folder_paths.get_output_directory()
    
    detected_checkpoint_clean = "Unknown"
    model_hash_v2 = "Unknown"
    hashes_dict = {}
    lora_hashes_list = []
    
    if prompt_data is not None:
        for node_id in prompt_data:
            node = prompt_data[node_id]
            class_type = node.get('class_type', '')
            inputs = node.get('inputs', {})
            
            # 1. RILEVAMENTO CHECKPOINT
            if class_type in ['CheckpointLoaderSimple', 'CheckpointLoader']:
                ckpt_path_rel = inputs.get('ckpt_name', '')
                if ckpt_path_rel:
                    detected_checkpoint_clean = clean_name(ckpt_path_rel)
                    full_path = folder_paths.get_full_path("checkpoints", ckpt_path_rel)
                    model_hash_v2 = get_sha256_hash(full_path)
                    hashes_dict["model"] = model_hash_v2
            
            # 2. RILEVAMENTO LORA (Nodi Standard)
            if class_type in ['LoraLoader', 'LoraLoaderModelOnly']:
                lora_path_rel = inputs.get('lora_name', '')
                if lora_path_rel:
                    l_name = clean_name(lora_path_rel)
                    l_path = folder_paths.get_full_path("loras", lora_path_rel)
                    l_hash = get_sha256_hash(l_path)
                    hashes_dict[f"lora:{l_name}"] = l_hash
                    lora_hashes_list.append(f"{l_name}: {l_hash}")

            # 3. RILEVAMENTO LORA (Tuo Nodo SuperPrompt Mixer)
            if class_type == 'FurreySuperPrompt':
                for i in range(1, 4):
                    l_name_raw = inputs.get(f'lora_{i}_name', 'None')
                    if l_name_raw and l_name_raw != "None":
                        l_name = clean_name(l_name_raw)
                        l_path = folder_paths.get_full_path("loras", l_name_raw)
                        l_hash = get_sha256_hash(l_path)
                        hashes_dict[f"lora:{l_name}"] = l_hash
                        lora_hashes_list.append(f"{l_name}: {l_hash}")

        # 4. LORA NEL TESTO (Regex per <lora:name:1.0>)
        all_text = f"{pos_text} {neg_text}"
        found_loras = re.findall(r"<lora:([^:>]+)(?::[^>]+)?>", all_text)
        available_loras = folder_paths.get_filename_list("loras")
        for l_raw in found_loras:
            l_name_with_ext = l_raw if l_raw.endswith(".safetensors") else f"{l_raw}.safetensors"
            if l_name_with_ext in available_loras:
                l_name = clean_name(l_name_with_ext)
                if f"lora:{l_name}" not in hashes_dict:
                    l_path = folder_paths.get_full_path("loras", l_name_with_ext)
                    l_hash = get_sha256_hash(l_path)
                    hashes_dict[f"lora:{l_name}"] = l_hash
                    lora_hashes_list.append(f"{l_name}: {l_hash}")

    final_model_name = clean_name(model_name) if (model_name != "" and model_name != "model.safetensors") else detected_checkpoint_clean

    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[2], images[0].shape[1])
    results = list()
    
    for image in images:
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = PngInfo()
        
        if mode in ["ComfyUI + A1111", "ComfyUI Only"]:
            if prompt_data is not None: metadata.add_text("prompt", json.dumps(prompt_data))
            if extra_pnginfo is not None:
                for x in extra_pnginfo: metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        if mode in ["ComfyUI + A1111", "A1111 Only"]:
            w, h = img.size
            s_name = sampler.replace('_', ' ').title()
            s_type = scheduler.title()
            
            a1111_str = f"{pos_text}\nNegative prompt: {neg_text}\n"
            a1111_str += f"Steps: {steps}, Sampler: {s_name}, Schedule type: {s_type}, CFG scale: {cfg}, Seed: {seed}, Size: {w}x{h}, "
            a1111_str += f"Model hash: {model_hash_v2}, Model: {final_model_name}, "
            if lora_hashes_list:
                a1111_str += f"Lora hashes: \"{', '.join(lora_hashes_list)}\", "
            a1111_str += f"Denoising strength: {denoise}, Hashes: {json.dumps(hashes_dict)}"
            
            metadata.add_text("parameters", a1111_str)

        if mode == "No Metadata": metadata = None

        file = f"{filename}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        counter += 1
    return results

# ==========================================
# CLASSI NODI (Furrey SuperTools)
# ==========================================

class FurreySimpleText:
    @classmethod
    def INPUT_TYPES(s): return {"required": {"text": ("STRING", {"multiline": True, "default": ""})}}
    RETURN_TYPES = ("STRING",); FUNCTION = "run"; CATEGORY = "Furrey/SuperTools"
    def run(self, text): return (text,)

class FurreyAllInOne:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), "vae": ("VAE",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                "width": ("INT", {"default": 1024}), "height": ("INT", {"default": 1024}),
                "positive_text": ("STRING", {"forceInput": True}), "negative_text": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20}), "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ), "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0}), "filename_prefix": ("STRING", {"default": "FurreyImg"}),
                "model_name_str": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ("LATENT",); FUNCTION = "run"; OUTPUT_NODE = True; CATEGORY = "Furrey/SuperTools"
    def run(self, **kwargs):
        latent = {"samples": torch.zeros([1, 4, kwargs['height'] // 8, kwargs['width'] // 8])}
        res = nodes.common_ksampler(kwargs['model'], kwargs['seed'], kwargs['steps'], kwargs['cfg'], kwargs['sampler_name'], kwargs['scheduler'], kwargs['positive'], kwargs['negative'], latent, kwargs['denoise'])[0]
        pix = kwargs['vae'].decode(res["samples"])
        save_furrey_logic(pix, kwargs['filename_prefix'], kwargs.get('prompt'), kwargs.get('extra_pnginfo'), kwargs['positive_text'], kwargs['negative_text'], kwargs['steps'], kwargs['sampler_name'], kwargs['scheduler'], kwargs['cfg'], kwargs['seed'], kwargs['model_name_str'], kwargs['denoise'], "ComfyUI + A1111")
        return {"ui": {"images": []}, "result": (res,)}

class FurreyHiresFix:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_base": ("LATENT",), "model": ("MODEL",), "vae": ("VAE",), "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                "upscale_by": ("FLOAT", {"default": 1.5}), "hires_denoise": ("FLOAT", {"default": 0.5}),
                "positive_text": ("STRING", {"forceInput": True}), "negative_text": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20}), "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ), "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "filename_prefix": ("STRING", {"default": "FurreyHires"}),
                "model_name_str": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = ("LATENT",); FUNCTION = "run"; OUTPUT_NODE = True; CATEGORY = "Furrey/SuperTools"
    def run(self, **kwargs):
        samples = kwargs['latent_base']["samples"]
        w, h = round(samples.shape[3] * kwargs['upscale_by']), round(samples.shape[2] * kwargs['upscale_by'])
        s = comfy.utils.common_upscale(samples, w, h, "nearest-exact", "center")
        res = nodes.common_ksampler(kwargs['model'], kwargs['seed'], kwargs['steps'], kwargs['cfg'], kwargs['sampler_name'], kwargs['scheduler'], kwargs['positive'], kwargs['negative'], {"samples": s}, denoise=kwargs['hires_denoise'])[0]
        pix = kwargs['vae'].decode(res["samples"])
        save_furrey_logic(pix, kwargs['filename_prefix'], kwargs.get('prompt'), kwargs.get('extra_pnginfo'), kwargs['positive_text'], kwargs['negative_text'], kwargs['steps'], kwargs['sampler_name'], kwargs['scheduler'], kwargs['cfg'], kwargs['seed'], kwargs['model_name_str'], kwargs['hires_denoise'], "ComfyUI + A1111")
        return {"ui": {"images": []}, "result": (res,)}

class FurreySaveImagePlus:
    def __init__(self): self.output_dir = folder_paths.get_output_directory()
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "FurreyPlus"}),
                "save_mode": (["ComfyUI + A1111", "A1111 Only", "ComfyUI Only", "No Metadata", "Preview Only"],),
                "positive": ("STRING", {"forceInput": True}),
                "negative": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "model_name": ("STRING", {"default": ""}),
                "denoise": ("FLOAT", {"default": 1.0}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }
    RETURN_TYPES = (); FUNCTION = "execute"; OUTPUT_NODE = True; CATEGORY = "Furrey/SuperTools"
    def execute(self, images, **kwargs):
        if kwargs['save_mode'] == "Preview Only":
            temp_dir = folder_paths.get_temp_directory()
            results = list()
            for image in images:
                i = 255. * image.cpu().numpy(); img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                file = f"preview_{np.random.randint(1000)}.png"; img.save(os.path.join(temp_dir, file))
                results.append({"filename": file, "subfolder": "", "type": "temp"})
            return {"ui": {"images": results}}
        results = save_furrey_logic(images, kwargs['filename_prefix'], kwargs.get('prompt'), kwargs.get('extra_pnginfo'), kwargs['positive'], kwargs['negative'], kwargs['steps'], kwargs['sampler_name'], kwargs['scheduler'], kwargs['cfg'], kwargs['seed'], kwargs['model_name'], kwargs['denoise'], kwargs['save_mode'])
        return {"ui": {"images": results}}

# ==========================================
# MAPPATURE
# ==========================================
NODE_CLASS_MAPPINGS = {
    "FurreySimpleText": FurreySimpleText,
    "FurreyAllInOne": FurreyAllInOne,
    "FurreyHiresFix": FurreyHiresFix,
    "FurreySaveImagePlus": FurreySaveImagePlus
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FurreySimpleText": "Furrey Text Box",
    "FurreyAllInOne": "Furrey KSampler & Save (Base)",
    "FurreyHiresFix": "Furrey Hires Fix & Save",
    "FurreySaveImagePlus": "Furrey Save Image Plus 🐾"
}