import logging
import time
from typing import List, Optional
import guidance
from omegaconf import DictConfig, OmegaConf
import stopit
from ..utils import (
    GENERATION_TIMEOUT,
    TokenUsage,
    GenerationMetadata,
    GenerationResponse,
    CompileStatus,
    CompileStatusCode,
    DecodingStatus,
    DecodingStatusCode,
    profile_generation,
)
from ._engine import BaseEngine
from ..config import GuidanceGenConfig


logger = logging.getLogger(__name__)

from guidance._parser import TokenParserException


class GuidanceEngine(BaseEngine):

    config_cls = GuidanceGenConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.engine == "hf":
            from transformers import BitsAndBytesConfig
            import torch

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
                self.guidance_model_state = guidance.models.Transformers(
                    cfg.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.guidance_model_state = guidance.models.Transformers(
                    cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto"
                )
                # block the echo
            self.guidance_model_state.echo = False
            self._max_ctx_len = (
                self.guidance_model_state.engine.tokenizer._orig_tokenizer.model_max_length
            )
        elif cfg.engine == "llamacpp":
            if cfg.model_cls == "Llama":
                from llama_cpp import Llama

                llamacpp_model = Llama.from_pretrained(
                    repo_id=cfg.repo_id, filename=cfg.filename, **cfg.kwargs
                )
            else:
                raise NotImplementedError
            self.guidance_model_state = guidance.models.LlamaCpp(
                llamacpp_model, echo=False
            )
            self._max_ctx_len = cfg.kwargs.get("n_ctx", 1024)
        else:
            raise NotImplementedError

    @staticmethod
    def init(cfg):
        return GuidanceEngine(cfg)

    def preprocess_prompts_before_run(self, prompts, generation_config):
        model_inputs = prompts
        return model_inputs

    def _call_engine(
        self, prompt, operator, stream=True, metadata: GenerationMetadata = None
    ):
        try:
            with stopit.ThreadingTimeout(GENERATION_TIMEOUT) as to_ctx_mgr:
                if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                    if stream:
                        state_iterator = (
                            self.guidance_model_state.stream() + prompt + operator
                        )
                        for i, state in enumerate(state_iterator):
                            if i == 0 and metadata:
                                metadata._first_tok_arr_time = time.time()
                        final_state = state
                    else:
                        final_state = self.guidance_model_state + prompt + operator
            if to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
                metadata.decoding_status = DecodingStatus(
                    code=DecodingStatusCode.DECODING_TIMEOUT,
                    message="Operation timed out",
                )
                final_state = self.guidance_model_state
        except BaseException as e:
            if "Attempted to use a context length of" in str(e) or isinstance(
                e, TokenParserException
            ):

                metadata.decoding_status = DecodingStatus(
                    code=DecodingStatusCode.EXCEEDING_MAX_CTX, message=str(e)
                )
                final_state = self.guidance_model_state
            else:
                metadata.decoding_status = DecodingStatus(
                    code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
                )
                final_state = self.guidance_model_state
        return final_state

    @staticmethod
    def compile_grammar(
        json_schema_obj, kwargs_wo_json_schema, metadata: GenerationMetadata = None
    ):
        """Extract and compile JSON schema from configuration."""

        # Convert JSON schema object if it's an OmegaConf DictConfig
        if isinstance(json_schema_obj, DictConfig):
            json_schema_obj = OmegaConf.to_container(json_schema_obj)

        # Compile the generation operation, handling schema support errors
        try:
            generation_op = guidance.json(
                schema=json_schema_obj,
                name="generated_object",
                **kwargs_wo_json_schema,
            )
            if metadata:
                metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
                metadata._grammar_compilation_end_time = time.time()
            return generation_op
        except BaseException as e:
            logger.warn(f"Error compiling JSON schema: {e}")
            if metadata:
                metadata.compile_status = CompileStatus(
                    code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
                )
            return None

    def _fetch_variable(
        self, variable_name: str, metadata: Optional[GenerationMetadata] = None
    ) -> str:
        if not metadata.is_valid_so_far():
            return None
        try:
            generation = self.guidance_model_state[variable_name]
            metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)
        except KeyError as e:
            # KeyError: "Model does not contain the variable 'generated_object'"
            # bug in guidance, but only happens very rarely, hard to reproduce
            # TODO: we should have another status code for this
            metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
            )
            generation = None
        return generation

    def _generate_free_mode(
        self,
        prompt: str,
        generation_config: GuidanceGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        stream: bool = generation_config.get("stream", False)
        generation_op = guidance.gen(
            **generation_config.operator.kwargs,
            name="generated_object",
        )  # max_tokens, stop, regex
        self.guidance_model_state = self._call_engine(
            prompt, generation_op, stream=stream, metadata=metadata
        )
        generation: Optional[str] = self._fetch_variable("generated_object", metadata)
        return generation

    def _generate_json_mode(
        self,
        prompt: str,
        generation_config: GuidanceGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        # just set the json schema to be None
        if "json_schema" in generation_config.operator.kwargs:
            generation_config.operator.kwargs["json_schema"] = None
        return self._generate_json_schema_mode(prompt, generation_config, metadata)

    def _generate_json_schema_mode(
        self,
        prompt: str,
        generation_config: GuidanceGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        # Load the JSON schema file
        json_schema_obj = generation_config.operator.kwargs.get("json_schema")

        stream: bool = generation_config.get("stream", False)

        # Filter out 'json_schema' from kwargs
        kwargs_wo_json_schema = {
            k: v
            for k, v in generation_config.operator.kwargs.items()
            if k != "json_schema"
        }
        # Extract and compile JSON schema from configuration
        generation_op = self.compile_grammar(
            json_schema_obj, kwargs_wo_json_schema, metadata
        )

        # if the grammar is compiled successfully, we can decode
        if metadata.compile_status.code == CompileStatusCode.OK:
            # try:
            self.guidance_model_state = self._call_engine(
                prompt, generation_op, stream=stream, metadata=metadata
            )
            generation: Optional[str] = self._fetch_variable(
                "generated_object", metadata
            )
        else:
            generation = None

        return generation

    @staticmethod
    def prepare_json_schema(kwargs_conf):
        pass

    @profile_generation
    def generate(
        self,
        prompt: str,
        generation_config: GuidanceGenConfig,
        schema: Optional[str] = None,
    ) -> GenerationResponse:
        if schema is not None:
            generation_config = generation_config.update_json_schema(schema)

        gen_metadata = GenerationMetadata()

        gen_config = generation_config.gen_config

        if gen_config.mode == "json_schema":
            generation = self._generate_json_schema_mode(
                prompt, gen_config, gen_metadata
            )
        elif gen_config.mode == "json_mode":
            generation = self._generate_json_mode(prompt, gen_config, gen_metadata)
        elif gen_config.mode == "free":
            generation = self._generate_free_mode(prompt, gen_config, gen_metadata)
        else:
            raise NotImplementedError

        usage = self.get_token_usage(prompt, generation)

        response = GenerationResponse(
            input=prompt,
            output=generation,
            generated_tokens=None,
            metadata=gen_metadata,
            perf_metrics=None,
            token_usage=usage,
        )
        response.set_stop_reason(
            gen_config.operator.get("kwargs").get("max_tokens", None)
        )
        # this removes the prefix, otherwise the next generation will continue
        self.guidance_model_state.reset()
        self.reset_token_usage()
        return response

    def get_token_usage(self, input: str, output: Optional[str]) -> TokenUsage:

        input_byte_string = input.encode("utf-8")
        num_input_tokens = len(
            self.guidance_model_state.engine.tokenizer.encode(input_byte_string)
        )

        if output is None or output == "":
            return TokenUsage(input_tokens=num_input_tokens)

        output_byte_string = output.encode("utf-8")
        num_output_tokens = len(
            self.guidance_model_state.engine.tokenizer.encode(output_byte_string)
        )

        usage = TokenUsage(
            input_tokens=num_input_tokens,
            output_tokens=num_output_tokens,
            no_ff_output_tokens=self.guidance_model_state.engine.metrics.engine_output_tokens,
            ff_output_tokens=num_output_tokens
            - self.guidance_model_state.engine.metrics.engine_output_tokens,
        )

        return usage

    def reset_token_usage(self):
        self.guidance_model_state.engine.metrics.engine_input_tokens = 0
        self.guidance_model_state.engine.metrics.engine_output_tokens = 0

    @property
    def max_ctx_len(self):
        return self._max_ctx_len

    def encode(self, text: str) -> List[int]:
        byte_string = text.encode("utf-8")
        return self.guidance_model_state.engine.tokenizer.encode(byte_string)

    def decode(self, ids: List[int]) -> str:
        byte_string = self.guidance_model_state.engine.tokenizer.decode(ids)
        return byte_string.decode("utf-8")
