from .furrey_nodes import FurreyAllInOne, FurreySimpleText, FurreyHiresFix

NODE_CLASS_MAPPINGS = {
    "FurreyAllInOne": FurreyAllInOne,
    "FurreySimpleText": FurreySimpleText,
    "FurreyHiresFix": FurreyHiresFix
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FurreyAllInOne": "Furrey KSampler & Save (Base)",
    "FurreySimpleText": "Furrey Text Box",
    "FurreyHiresFix": "Furrey Hires Fix & Save"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]