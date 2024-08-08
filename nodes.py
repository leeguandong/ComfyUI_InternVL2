import os

import folder_paths
import comfy.model_management as mm

import io
import base64
import torch
import requests
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from typing import Union, List

# from lmdeploy import TurbomindEngineConfig, pipeline
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel


class InternVLModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "OpenGVLab/InternVL2-1B",
                        "OpenGVLab/InternVL2-2B",
                        "OpenGVLab/InternVL2-4B",
                        "OpenGVLab/InternVL2-8B",
                        "OpenGVLab/InternVL2-26B",
                        "OpenGVLab/InternVL2-40B",
                    ],
                    {
                        "default": "OpenGVLab/InternVL2-2B"
                    }),
            }
        }

    RETURN_TYPES = ("InternVLModel",)
    RETURN_NAMES = ("intervl_model",)
    FUNCTION = "load_model"
    CATEGORY = "internvl"

    def load_model(self, model):
        device = mm.get_torch_device()

        model_name = model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "LLM", model_name))

        if not os.path.exists(model_dir):
            print(f"Downloading {model}")
            snapshot_download(repo_id=model, cache_dir=model_dir, local_dir_use_symlinks=False)
            # huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-2B --local-dir InternVL2-2B

        tokenizer = AutoTokenizer.from_pretrained(model_dir,trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir,
                                          torch_dtype=torch.float16,
                                          trust_remote_code=True,
                                          low_cpu_mem_usage=True).eval().to(device)

        model = {
            "model": model,
            "tokenizer": tokenizer
        }
        return (model,)


class DynamicPreprocess:
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "min_num": ("INT", {"default": 1, "min": 1, "max": 40}),
                "max_num": ("INT", {"default": 6, "min": 1, "max": 40}),
                "image_size": ("INT", {"default": 448, }),
                "use_thumbnail": ("BOOLEAN", {"default": True, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    CATEGORY = "internvl"

    def load_image(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        pil_image = self.convert_to_pil_image(image)
        transform = self.build_transform(input_size=image_size)
        images = self.preprocess(pil_image, min_num, max_num, image_size, use_thumbnail)
        # import pdb;pdb.set_trace()
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return (pixel_values,)

    def preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # 该代码功能是生成并排序一个集合，其中包含所有在指定范围内（min_num和max_num）的两个数的乘积
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # 寻找与给定aspect_ratio最接近的宽高比
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def convert_to_pil_image(self, image: Union[
        np.ndarray, List[np.ndarray], bytes, str, Image.Image, torch.Tensor]) -> Image.Image:

        try:
            if isinstance(image, np.ndarray):
                return Image.fromarray(self._ensure_rgb(image))

            elif isinstance(image, list):
                return self._handle_list_input(image)

            elif isinstance(image, bytes):
                return Image.open(io.BytesIO(image)).convert('RGB')

            elif isinstance(image, str):
                return self._handle_string_input(image)

            elif isinstance(image, Image.Image):
                return image.convert('RGB')

            elif isinstance(image, torch.Tensor):
                return self._convert_tensor_to_pil(image)

            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")

    def _handle_list_input(self, image_list: List) -> Image.Image:
        if len(image_list) == 0:
            raise ValueError("Empty list provided as image")

        if isinstance(image_list[0], np.ndarray):
            return Image.fromarray(self._ensure_rgb(image_list[0]))

        elif all(isinstance(x, (int, float)) for x in image_list):
            arr = np.array(image_list).astype('uint8')

            if arr.size in [1024 * 1024, 1024 * 1024 * 3]:
                arr = arr.reshape((1024, 1024, -1))
            elif arr.size in [512 * 512, 512 * 512 * 3]:
                arr = arr.reshape((512, 512, -1))
            else:
                arr = arr.reshape((arr.shape[0], -1))
            return Image.fromarray(self._ensure_rgb(arr))

        else:
            raise ValueError(f"Unsupported list content type: {type(image_list[0])}")

    def _handle_string_input(self, image_string: str) -> Image.Image:
        if image_string.startswith(('http://', 'https://')):
            response = requests.get(image_string)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content)).convert('RGB')

        elif image_string.startswith('data:image'):
            image_data = base64.b64decode(image_string.split(',')[1])
            return Image.open(io.BytesIO(image_data)).convert('RGB')

        else:
            return Image.open(image_string).convert('RGB')

    def _ensure_rgb(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            return np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            return arr
        elif arr.ndim == 3 and arr.shape[2] == 4:
            return arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

    def _convert_tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0)
        elif tensor.ndim == 2:

            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)

        np_array = tensor.cpu().numpy()

        if np_array.dtype != np.uint8:
            if np_array.max() <= 1.0:
                np_array = (np_array * 255).astype(np.uint8)
            else:
                np_array = np_array.astype(np.uint8)

        return Image.fromarray(self._ensure_rgb(np_array))

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform


class InternVLHFInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("InternVLModel",),
                "system_prompt": ("STRING", {
                    "multiline": False,
                    "default": "You are a helpful assistant."
                }),
                "prompt": ("STRING", {
                    "multiline": False,
                    "default": "What is this?"
                }),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "do_sample": ("BOOLEAN", {"default": False}),
                "num_beams": ("INT", {"default": 1})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "internvl"

    def process(self,
                image,
                model,
                system_prompt,
                prompt,
                keep_model_loaded=False,
                max_new_tokens=1024,
                do_sample=False,
                num_beams=1):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        image = image.to(torch.float16).to(device)
        generation_config = dict(num_beams=num_beams, max_new_tokens=max_new_tokens, do_sample=do_sample)

        internvl_model = model['model']
        tokenizer = model['tokenizer']
        question = f'<image>\n{system_prompt}\n{prompt}'
        response, _ = internvl_model.chat(tokenizer, image, question, generation_config, history=None,
                                 return_history=True)

        if not keep_model_loaded:
            print("Offloading model...")
            internvl_model.to(offload_device)
            mm.soft_empty_cache()

        return (response,)


class InternVLLMDEPLOYInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (
                    [
                        "OpenGVLab/InternVL2-1B",
                        "OpenGVLab/InternVL2-2B",
                        "OpenGVLab/InternVL2-4B",
                        "OpenGVLab/InternVL2-8B",
                        "OpenGVLab/InternVL2-26B",
                        "OpenGVLab/InternVL2-40B",
                    ],
                    {
                        "default": "OpenGVLab/InternVL2-2B"
                    }),
                "system_prompt": ("STRING", {
                    "multiline": False,
                    "default": "You are a helpful assistant."
                }),
                "prompt": ("STRING", {
                    "multiline": False,
                    "default": "What is this?"
                }),
            },
            "optional": {
                "tp": ("INT", {"default": 1, "min": 1, "max": 8}),
                "cache_max_entry_count": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "internvl"

    def process(self,
                image,
                model,
                system_prompt,
                prompt,
                tp=1,
                cache_max_entry_count=0.5):
        mm.soft_empty_cache()
        # load_model
        model_name = model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "LLM", model_name))

        if not os.path.exists(model_dir):
            print(f"Downloading {model}")
            snapshot_download(repo_id=model, cache_dir=model_dir, local_dir_use_symlinks=False)

        pipe = pipeline(model_dir,
                        backend_config=TurbomindEngineConfig(tp=tp, cache_max_entry_count=cache_max_entry_count))
        image_pil = self.tensor2pil(image).convert("RGB")
        question = f'<image>\n{system_prompt}\n{prompt}'
        response = pipe((question, image_pil))
        return (response,)

    def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


NODE_CLASS_MAPPINGS = {
    "InternVLModelLoader": InternVLModelLoader,
    "DynamicPreprocess": DynamicPreprocess,
    "InternVLHFInference": InternVLHFInference,
    # "InternVLSWIFTInference",
    # "InternVLLMDEPLOYInference": InternVLLMDEPLOYInference
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InternVLModelLoader": "InternVL Model Loader",
    "DynamicPreprocess": "Dynamic Preprocess",
    "InternVLHFInference": "InternVL HF Inference",
    # "InternVLLMDEPLOYInference": "InternVL LMDEPLOY Inference"
}
