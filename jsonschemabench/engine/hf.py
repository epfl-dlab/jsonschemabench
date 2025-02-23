import logging
from typing import List, Optional
from ..utils import (
    Token,
    TokenUsage,
    GenerationResponse,
    profile_generation,
)
from ._engine import BaseEngine
from ..config import HFGenConfig


logger = logging.getLogger(__name__)


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.utils import ModelOutput


from typing import Dict, List, Optional


class HFEngine(BaseEngine):

    config_cls = HFGenConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        # Load the model with bitsandbytes if 8bit or 4bit flag is set
        if cfg.get("use_8bit", False) or cfg.get("use_4bit", False):
            try:
                pass
            except ImportError:
                raise ImportError(
                    "You need to install bitsandbytes to use 8-bit or 4-bit modes. Install it with `pip install bitsandbytes`."
                )

            bnb_config = BitsAndBytesConfig(
                load_in_8bit=cfg.get("use_8bit", False),
                load_in_4bit=cfg.get("use_4bit", False),
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                cfg.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            self.hf_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                cfg.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
            )

        self.hf_tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True
        )
        self._max_ctx_len = self.hf_tokenizer.model_max_length
        self.hf_tokenizer.pad_token = self.hf_tokenizer.eos_token

    @staticmethod
    def init(cfg):
        return HFEngine(cfg)

    @profile_generation
    def generate(
        self, prompt: str, generation_config: HFGenConfig, schema: Optional[str] = None
    ) -> GenerationResponse:
        if schema is not None:
            generation_config = generation_config.update_json_schema(schema)
        # Generate text from the model
        model_input: Dict[str, torch.Tensor] = self.hf_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )

        # print the input tensor shape
        logger.info(
            f"Input tensor shape: {model_input['input_ids'].shape}, where batch_size={model_input['input_ids'].shape[0]}, padded_seq_len={model_input['input_ids'].shape[1]}"
        )

        model_output: ModelOutput = self.hf_model.generate(
            model_input["input_ids"],
            generation_config=generation_config.gen_config,
            attention_mask=model_input["attention_mask"],
            tokenizer=self.hf_tokenizer,
        )

        # here we need to remove the tokens that are from input_ids, we only want the generated tokens
        input_length = model_input["input_ids"].shape[1]
        pure_output_sequences = model_output.sequences[:, input_length:]
        output_texts: List[str] = self.hf_tokenizer.batch_decode(
            pure_output_sequences, skip_special_tokens=True
        )

        # input batch size may be different from the output batch size due to beam search
        # or best n sampling
        input_batch_size = model_input["input_ids"].shape[0]
        output_batch_size = model_output.sequences.shape[0]
        # assert the ratio is integer
        num_gens_per_sample = output_batch_size // input_batch_size
        assert (
            output_batch_size % input_batch_size == 0
        ), f"output_batch_size: {output_batch_size}, input_batch_size: {input_batch_size}, the ratio of them should be an integer"

        best_texts: list[str] = self.pick_best_generation(
            output_texts,
            input_batch_size=input_batch_size,
            num_gens_per_sample=num_gens_per_sample,
        )

        gen_response = GenerationResponse(
            input=prompt,
            output=best_texts,
            generated_tokens=[
                Token(id=self.convert_token_to_id(token_str), text=token_str)
                for token_str in output_texts
            ],
            metadata=None,
        )
        return gen_response

    def encode(self, text: str) -> List[int]:
        return self.hf_tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.hf_tokenizer.decode(ids)

    @property
    def max_ctx_len(self):
        return self._max_ctx_len

    def get_token_usage(
        self, model_input, model_output: Optional[ModelOutput] = None
    ) -> TokenUsage:
        # TODO: this only works for batch size 1
        # for larger batch sizes, the padding should be taken into account
        # return {
        #     "input_tokens": model_input["input_ids"].shape[1],
        #     "output_tokens": model_output.sequences.shape[1],
        # }
        input_tokens = model_input["input_ids"].shape[1]
        input_and_output_tokens = (
            model_output.sequences.shape[1] if model_output is not None else None
        )
        pure_output_tokens = (
            input_and_output_tokens - input_tokens
            if input_and_output_tokens is not None
            else 0
        )
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=pure_output_tokens,
        )


    @staticmethod
    def pick_best_generation(
        generations: List[str],
        input_batch_size: int,
        num_gens_per_sample: int,
        strategy: str = "first",
    ) -> List[str]:

        batches: List[List[str]] = split_batch_into_batches(
            generations,
            input_batch_size=input_batch_size,
            num_gens_per_sample=num_gens_per_sample,
        )
        if strategy == "first":
            return batch_get_first_generation(batches)
        else:
            raise ValueError(f"Strategy {strategy} not supported")


def split_batch_into_batches(
    generations: List[str],
    input_batch_size: Optional[int] = None,
    num_gens_per_sample: Optional[int] = None,
) -> List[List[str]]:
    """Split a list of generations into batches of size bs

    c.f. https://huggingface.co/docs/transformers/v4.19.2/en/internal/generation_utils#transformers.generation_utils.BeamSearchDecoderOnlyOutput

    Huggingface's Transformers library uses a flattened shape for the outputs of beam search.
    This function generates a flattened list of generations.
    By flattening it into a single dimension, it simplifies the logic and handling in certain parts of the codebase.

    If you want to match outputs to their corresponding inputs or if you're handling post-processing, having the outputs
    in the shape (batch_input_size,num_beam,max_seq_len) can be more intuitive.

    """
    if input_batch_size is None and num_gens_per_sample is None:
        raise ValueError("Either bs or num_beams must be provided")

    if input_batch_size is not None and num_gens_per_sample is not None:
        assert (
            len(generations) == input_batch_size * num_gens_per_sample
        ), f"len(generations)={len(generations)} but bs={input_batch_size} and num_beams={num_gens_per_sample}"

    if num_gens_per_sample is None:
        num_gens_per_sample = len(generations) // input_batch_size
    return [
        generations[i : i + num_gens_per_sample]
        for i in range(0, len(generations), num_gens_per_sample)
    ]


def batch_get_first_generation(output_texts: List[List[str]]) -> List[str]:
    results: List[str] = [texts[0] for texts in output_texts]
    return results
