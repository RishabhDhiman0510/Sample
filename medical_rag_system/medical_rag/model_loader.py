"""ModelLoader abstraction for loading PEFT models with quantization."""

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

from .utils.logging import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Load and manage LLM with PEFT adapters and quantization."""

    def __init__(
        self,
        adapter_path: str,
        base_model_path: Optional[str] = None,
        load_in_8bit: bool = True,
        device_map: str = "auto"
    ):
        self.adapter_path = adapter_path
        self.base_model_path = base_model_path
        self.load_in_8bit = load_in_8bit
        self.device_map = device_map
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("model_loader_initialized", adapter_path=adapter_path, load_in_8bit=load_in_8bit, device=self.device)

    def load(self) -> tuple:
        logger.info("loading_model_start")
        try:
            peft_config = PeftConfig.from_pretrained(self.adapter_path)
            base_model_name = self.base_model_path or peft_config.base_model_name_or_path
            logger.info("base_model_identified", base_model=base_model_name)

            quantization_config = None
            if self.load_in_8bit and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=False)
                logger.info("8bit_quantization_enabled")

            logger.info("loading_base_model")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name, quantization_config=quantization_config,
                device_map=self.device_map if torch.cuda.is_available() else None,
                trust_remote_code=True, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            logger.info("loading_peft_adapter")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("model_loaded_successfully")
            return self.model, self.tokenizer
        except Exception as e:
            logger.error("model_load_error", error=str(e))
            raise

    def generate(self, prompt: str, max_new_tokens: int = 120, temperature: float = 0.6, top_p: float = 0.9, repetition_penalty: float = 1.3, do_sample: bool = True) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample, pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=repetition_penalty, no_repeat_ngram_size=3)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
