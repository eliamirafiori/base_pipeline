from typing import List, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class SmallVLMEvaluator:
    """
    Ultra-lightweight VLM wrapper using SmolVLM2-500M-Video-Instruct.
    Configured specifically for 6 GB Maxwell GPUs (GTX 980 Ti) using float16.
    """

    def __init__(self, model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
        print(f"Loading lightweight VLM: {model_id} ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_id)
        # Disable multi-crop tiling so 16 frames stay within ~1.3k tokens and <2.5GB VRAM
        if hasattr(self.processor, "image_processor"):
            self.processor.image_processor.do_image_splitting = False

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

    def build_fixed_prompt(self, question: str, options: Optional[List[str]]) -> str:
        if options:
            opts_str = "\n".join(options)
            return (
                "You are given keyframes sampled in chronological order from a video, "
                "each labeled with its timestamp.\n"
                f"Question: {question}\n"
                f"Options:\n{opts_str}\n"
                "Respond ONLY with the single capital letter of the correct option (A, B, C, or D)."
            )
        return (
            "You are given keyframes sampled in chronological order from a video, "
            "each labeled with its timestamp.\n"
            f"Question: {question}\n"
            "Respond ONLY with a single numerical value (integer or decimal). Do not add units or explanation."
        )

    @torch.inference_mode()
    def answer_question(
        self,
        frames_with_ts: List[Tuple[str, Image.Image]],
        question: str,
        options: Optional[List[str]] = None,
    ) -> str:
        content = [{"type": "text", "text": "Keyframes in chronological order:"}]
        for ts, pil_img in frames_with_ts:
            content.append({"type": "text", "text": f"Frame [{ts}]:"})
            content.append({"type": "image", "image": pil_img})

        content.append(
            {"type": "text", "text": f"\n{self.build_fixed_prompt(question, options)}"}
        )
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: (
                v.to(self.device, dtype=self.dtype)
                if v.is_floating_point()
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }

        generated_ids = self.model.generate(
            **inputs, max_new_tokens=16, do_sample=False
        )
        prompt_len = inputs["input_ids"].shape[1]
        response = self.processor.batch_decode(
            generated_ids[:, prompt_len:], skip_special_tokens=True
        )[0]
        return response.strip()
