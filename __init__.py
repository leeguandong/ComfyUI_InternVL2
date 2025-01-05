from .nodes import InternVLModelLoader, DynamicPreprocess, InternVLHFInference

NODE_CLASS_MAPPINGS = {
    "InternVLModelLoader": InternVLModelLoader,
    "DynamicPreprocess": DynamicPreprocess,
    "InternVLHFInference": InternVLHFInference,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InternVLModelLoader": "InternVL Model Loader",
    "DynamicPreprocess": "Dynamic Preprocess",
    "InternVLHFInference": "InternVL HF Inference",
}

# Debugging: Print to confirm correct loading
print("[*] NODE_CLASS_MAPPINGS:", NODE_CLASS_MAPPINGS)
print("[*] NODE_DISPLAY_NAME_MAPPINGS:", NODE_DISPLAY_NAME_MAPPINGS)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

