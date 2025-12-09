import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import comfy.sd
import nodes
import comfy.model_management

# --- NODO 1: Casella di Testo Semplice ---
class FurreySimpleText:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "output_text"
    CATEGORY = "Furrey/SuperTools"

    def output_text(self, text):
        return (text,)

# --- Funzione di supporto per salvare le immagini (usata da entrambi i nodi) ---
def save_furrey_image(images, filename_prefix, output_dir, prompt, extra_pnginfo, pos_text, neg_text, steps, sampler, scheduler, cfg, seed, model_name, width, height, denoise):
    full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0])
    results = list()
    
    for image in images:
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        w, h = img.size
        pos_clean = pos_text.strip()
        neg_clean = neg_text.strip()
        
        # Formato A1111
        a1111_text = f"{pos_clean}\nNegative prompt: {neg_clean}\n"
        a1111_text += f"Steps: {steps}, Sampler: {sampler} {scheduler}, CFG scale: {cfg}, Seed: {seed}, Size: {w}x{h}, Model: {model_name}, Denoise: {denoise}"
        
        metadata.add_text("parameters", a1111_text)

        file = f"{filename}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": "output"
        })
        counter += 1
    return results

# --- NODO 2: All-in-One Generator ---
class FurreyAllInOne:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                
                "NOTICE": ("STRING", {"default": "⚠️ ANTEPRIMA BASSA QUALITÀ? È NORMALE! APRI IL FILE SALVATO. ⚠️", "multiline": True}),
                
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                
                "positive_text": ("STRING", {"multiline": True, "forceInput": True}),
                "negative_text": ("STRING", {"multiline": True, "forceInput": True}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                "filename_prefix": ("STRING", {"default": "FurreyImg"}),
                "model_name_str": ("STRING", {"default": "checkpoint.safetensors"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample_decode_save"
    OUTPUT_NODE = True
    CATEGORY = "Furrey/SuperTools"

    def sample_decode_save(self, model, vae, positive, negative, NOTICE, width, height, batch_size, positive_text, negative_text, seed, steps, cfg, sampler_name, scheduler, denoise, filename_prefix, model_name_str, prompt=None, extra_pnginfo=None):
        
        device = comfy.model_management.get_torch_device()
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=device)
        latent_image = {"samples": latent}

        latent_result = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]
        
        samples = latent_result["samples"]
        pixel_images = vae.decode(samples)

        results = save_furrey_image(pixel_images, filename_prefix, self.output_dir, prompt, extra_pnginfo, positive_text, negative_text, steps, sampler_name, scheduler, cfg, seed, model_name_str, width, height, denoise)

        return {"ui": {"images": results}, "result": (latent_result,)}

# --- NODO 3: Hires Fix / Refiner ---
class FurreyHiresFix:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent_base": ("LATENT",), # Output del nodo precedente
                "model": ("MODEL",),        # Puoi usare lo stesso modello o un Refiner
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                
                "NOTICE": ("STRING", {"default": "🔧 HIRES FIX: Ingrandisce e migliora i dettagli.", "multiline": True}),

                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.1}),
                "hires_denoise": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                
                "positive_text": ("STRING", {"multiline": True, "forceInput": True}),
                "negative_text": ("STRING", {"multiline": True, "forceInput": True}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                
                "filename_prefix": ("STRING", {"default": "FurreyHires"}),
                "model_name_str": ("STRING", {"default": "checkpoint.safetensors"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale_refine_save"
    OUTPUT_NODE = True
    CATEGORY = "Furrey/SuperTools"

    def upscale_refine_save(self, latent_base, model, vae, positive, negative, NOTICE, upscale_by, hires_denoise, positive_text, negative_text, seed, steps, cfg, sampler_name, scheduler, filename_prefix, model_name_str, prompt=None, extra_pnginfo=None):
        
        # 1. Upscale del Latente
        samples = latent_base["samples"]
        width = round(samples.shape[3] * upscale_by)
        height = round(samples.shape[2] * upscale_by)
        
        # Metodo di upscale ottimizzato per latenti (nearest-exact è standard per SDXL latent)
        s = comfy.utils.common_upscale(samples, width, height, "nearest-exact", "center")
        latent_upscaled = {"samples": s}

        # 2. Secondo KSampler (Refiner / Hires Fix)
        # Nota: usiamo hires_denoise qui
        latent_result = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_upscaled, denoise=hires_denoise)[0]

        # 3. Decode
        pixel_images = vae.decode(latent_result["samples"])

        # 4. Save
        # Calcoliamo la dimensione finale in pixel (x8 del latente)
        final_w = width * 8
        final_h = height * 8
        
        results = save_furrey_image(pixel_images, filename_prefix, self.output_dir, prompt, extra_pnginfo, positive_text, negative_text, steps, sampler_name, scheduler, cfg, seed, model_name_str, final_w, final_h, hires_denoise)

        return {"ui": {"images": results}, "result": (latent_result,)}