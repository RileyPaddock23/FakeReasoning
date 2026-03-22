import argparse
import io
import os
import re
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class FakeReasoningModule:
    def __init__(
        self,
        model_path: str,
        model_base: Optional[str] = None,
        conv_mode: str = "llava_v1",
        temperature: float = 0.2,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 512,
        device: str = "cuda",
    ) -> None:
        self.conv_mode = conv_mode
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        import torch
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        self._torch = torch
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            device=device,
        )

    def _build_prompt(self) -> str:
        from llava.constants import (
            DEFAULT_IMAGE_TOKEN,
            DEFAULT_IM_END_TOKEN,
            DEFAULT_IM_START_TOKEN,
            IMAGE_PLACEHOLDER,
        )
        from llava.conversation import conv_templates

        qs = (
            "Is this image real or fake? Please describe the image, reasoning step-by-step and conclude "
            "with 'this image is real' or 'this image is fake'. "
            "Respond in the following format: <REASONING>...</REASONING><CONCLUSION>...</CONCLUSION>."
        )

        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    @staticmethod
    def _parse_output(output_text: str) -> Dict[str, str]:
        reasoning_match = re.search(r"<REASONING>(.*?)</REASONING>", output_text, re.DOTALL)
        conclusion_match = re.search(r"<CONCLUSION>(.*?)</CONCLUSION>", output_text, re.DOTALL)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else output_text.strip()
        conclusion = conclusion_match.group(1).strip() if conclusion_match else output_text.strip()

        conclusion_lower = conclusion.lower()
        if "this image is fake" in conclusion_lower:
            label = "fake"
        elif "this image is real" in conclusion_lower:
            label = "real"
        else:
            full_lower = output_text.lower()
            if "this image is fake" in full_lower:
                label = "fake"
            elif "this image is real" in full_lower:
                label = "real"
            else:
                label = "unknown"

        return {
            "result": label,
            "reasoning": reasoning,
            "conclusion": conclusion,
            "raw_output": output_text.strip(),
        }

    def infer_image(self, image: Image.Image) -> Dict[str, str]:
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images, tokenizer_image_token

        prompt = self._build_prompt()
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).to(self.model.device)

        image_dtype = (
            self._torch.float16 if self.model.device.type == "cuda" else self._torch.float32
        )
        image_tensor = process_images([image], self.image_processor, self.model.config).to(
            self.model.device, dtype=image_dtype
        )

        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=None,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return self._parse_output(output_text)


def create_app(module: FakeReasoningModule) -> FastAPI:
    app = FastAPI(title="FakeReasoning API", version="1.0.0")

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded image exceeds size limit ({MAX_UPLOAD_BYTES} bytes).",
            )

        try:
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

        return module.infer_image(image)

    return app


_global_module: Optional[FakeReasoningModule] = None


def _read_env_float(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid float value for environment variable {name}: '{raw}'. Expected a numeric value."
        ) from exc


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid integer value for environment variable {name}: '{raw}'. Expected a whole number."
        ) from exc


def _get_global_app() -> FastAPI:
    global _global_module
    model_path = os.environ.get("FAKEREASONING_MODEL_PATH")
    if not model_path:
        app = FastAPI(title="FakeReasoning API", version="1.0.0")

        @app.get("/")
        async def root() -> Dict[str, str]:
            return {
                "message": "Set FAKEREASONING_MODEL_PATH (and optionally related FAKEREASONING_* vars) before using /predict."
            }

        return app

    if _global_module is None:
        _global_module = FakeReasoningModule(
            model_path=model_path,
            model_base=os.environ.get("FAKEREASONING_MODEL_BASE"),
            conv_mode=os.environ.get("FAKEREASONING_CONV_MODE", "llava_v1"),
            temperature=_read_env_float("FAKEREASONING_TEMPERATURE", 0.2),
            top_p=_read_env_float("FAKEREASONING_TOP_P", None),
            num_beams=_read_env_int("FAKEREASONING_NUM_BEAMS", 1),
            max_new_tokens=_read_env_int("FAKEREASONING_MAX_NEW_TOKENS", 512),
            device=os.environ.get("FAKEREASONING_DEVICE", "cuda"),
        )
    return create_app(_global_module)


app = _get_global_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    module = FakeReasoningModule(
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    app = create_app(module)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
