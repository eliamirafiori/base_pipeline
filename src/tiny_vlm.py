from typing import List, Optional, Tuple
from PIL import Image
import torch
from transformers import BlipForQuestionAnswering, BlipProcessor

from kfs_selectors import build_temporal_montage


class TinyBLIPVQA:
    """
    385M-parameter VLM compatible with torch==1.13.1 and 6GB GTX 980 Ti (<1.5 GB VRAM).
    """

    def __init__(self, model_id: str = "Salesforce/blip-vqa-base"):
        print(f"Loading {model_id} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(
            model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def build_prompt(self, question: str, options: Optional[List[str]]) -> str:
        if options:
            opts_text = " ".join(options)
            return (
                f"Chronological video frames with timestamps. {question} "
                f"Choices: {opts_text}. Answer with the option letter or exact choice:"
            )
        return f"Chronological video frames with timestamps. {question} Answer with a number:"

    @torch.inference_mode()
    def answer(
        self,
        keyframes: List[Tuple[str, Image.Image]],
        question: str,
        options: Optional[List[str]] = None,
    ) -> str:
        montage = build_temporal_montage(keyframes)
        prompt = self.build_prompt(question, options)

        inputs = self.processor(images=montage, text=prompt, return_tensors="pt")
        inputs = {
            k: (
                v.to(self.device, dtype=self.dtype)
                if v.is_floating_point()
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }

        out_ids = self.model.generate(**inputs, max_new_tokens=12)
        return self.processor.decode(out_ids[0], skip_special_tokens=True).strip()
