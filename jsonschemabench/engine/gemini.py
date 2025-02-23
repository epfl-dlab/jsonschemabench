import json
import logging
import os
import time
from typing import Optional
import google.generativeai as genai
from omegaconf import DictConfig, OmegaConf
from validation import is_json_schema_valid
from ..utils import (
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
from ..config import GeminiGenConfig


logger = logging.getLogger(__name__)

# global variable for schema check
model = None


class GeminiEngine(BaseEngine):

    config_cls = GeminiGenConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self.model = genai.GenerativeModel(self.cfg.model_name)  # 'gemini-1.5-flash'

    @staticmethod
    def init(cfg):
        return GeminiEngine(cfg)

    def _generate_free_mode(
        self,
        prompt: str,
        generation_config: GeminiGenConfig,
        metadata: GenerationMetadata,
    ) -> GenerationResponse:
        response: GenerationResponse = self._call_platform(
            prompt, generation_config, metadata
        )
        return response

    def _generate_json_mode(
        self,
        prompt: str,
        generation_config: GeminiGenConfig,
        metadata: GenerationMetadata,
    ) -> GenerationResponse:
        response: GenerationResponse = self._call_platform(
            prompt, generation_config, metadata
        )
        return response

    def _generate_json_schema_mode(
        self,
        prompt: str,
        generation_config: GeminiGenConfig,
        metadata: GenerationMetadata,
    ) -> GenerationResponse:

        response: GenerationResponse = self._call_platform(
            prompt, generation_config, metadata
        )
        return response

    @staticmethod
    def prepare_json_schema(kwargs_conf):
        pass

    def _call_platform(
        self,
        prompt: str,
        generation_config: GeminiGenConfig,
        metadata: GenerationMetadata,
    ):

        kwargs_dict = self.prepare_kwargs(generation_config.get("kwargs"))

        stream: bool = kwargs_dict.pop("stream", True)

        try:
            config = genai.types.GenerationConfig(**kwargs_dict)

            gemini_generator = self.model.generate_content(
                contents=prompt, generation_config=config, stream=stream
            )

            if stream:
                response = self.get_gemini_stream_response(gemini_generator, metadata)
            else:
                response = self.get_gemini_response(gemini_generator, metadata)
        except Exception as e:
            error_message = str(e)
            logger.warn(f"Error message: {error_message}")
            # here we assume all errors are due to unsupported schema
            # is this a good assumption? TODO

            metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=error_message
            )
            response = GenerationResponse(input=prompt, output="", metadata=metadata)

        response.input = prompt

        return response

    @profile_generation
    def generate(
        self,
        prompt: str,
        generation_config: GeminiGenConfig,
        schema: Optional[str] = None,
    ) -> GenerationResponse:
        if schema is not None:
            generation_config = generation_config.update_json_schema(schema)

        gen_metadata = GenerationMetadata()

        gen_config = generation_config.gen_config

        if gen_config.mode == "json_schema":
            response: GenerationResponse = self._generate_json_schema_mode(
                prompt, gen_config, gen_metadata
            )
        elif gen_config.mode == "json_mode":
            response: GenerationResponse = self._generate_json_mode(
                prompt, gen_config, gen_metadata
            )
        elif gen_config.mode == "free":
            response: GenerationResponse = self._generate_free_mode(
                prompt, gen_config, gen_metadata
            )
        else:
            raise NotImplementedError

        response.set_stop_reason(
            gen_config.get("kwargs").get("max_output_tokens", None)
        )

        return response

    def update_total_usage(self, gemini_response: GenerationResponse) -> None:
        self.total_usage["num_samples"] += 1
        self.total_usage["input_tokens"] += gemini_response.token_usage.input_tokens
        self.total_usage["output_tokens"] += gemini_response.token_usage.output_tokens

    def get_gemini_stream_response(
        self,
        gemini_generator: genai.types.GenerateContentResponse,
        metadata: GenerationMetadata = None,
    ) -> GenerationResponse:
        generation = ""

        for i, chunk in enumerate(gemini_generator):
            # print("chunk", chunk)
            if i == 0 and metadata:
                metadata._first_tok_arr_time = time.time()

            try:
                generation += chunk.text
            except Exception as e:
                logger.error(f"Error getting gemini response: {e}")
                if metadata:
                    metadata.decoding_status = DecodingStatus(
                        code=DecodingStatusCode.BAD_API_RESPONSE, message=str(e)
                    )
                break
        if metadata.is_valid_so_far():
            metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
            metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)

        usage: TokenUsage = self.get_token_usage(gemini_generator)

        response = GenerationResponse(
            input="",  # needs to be updated outside of this function
            output=generation,
            generated_tokens=None,
            token_usage=usage,
            metadata=metadata,
        )

        return response

    def get_gemini_response(
        self,
        gemini_response: genai.types.GenerateContentResponse,
        metadata: GenerationMetadata = None,
    ) -> GenerationResponse:
        output = gemini_response.text

        usage: TokenUsage = self.get_token_usage(gemini_response)

        if metadata.is_valid_so_far():
            metadata.compile_status = CompileStatus(code=CompileStatusCode.OK)
            metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)

        response = GenerationResponse(
            input="",  # needs to be updated outside of this function
            output=output,
            generated_tokens=None,
            token_usage=usage,
            metadata=metadata,
        )

        return response

    @staticmethod
    def get_token_usage(
        gemini_generator: genai.types.GenerateContentResponse,
    ) -> TokenUsage:
        usage = TokenUsage(
            input_tokens=gemini_generator.usage_metadata.prompt_token_count,
            output_tokens=gemini_generator.usage_metadata.candidates_token_count,
        )
        return usage

    def prepare_kwargs(self, kwargs_conf):
        # convert kwargs to a dictionary if it is still OmegaConf
        if isinstance(kwargs_conf, DictConfig):
            kwargs_dict = OmegaConf.to_container(kwargs_conf)
        else:
            assert (
                type(kwargs_conf) == dict
            ), f"kwargs_conf should be a dictionary if not an OmegaConf object, but got {type(kwargs_conf)}"
            kwargs_dict = kwargs_conf if kwargs_conf else {}

        # the response_schema should not be set for unconstrained generation, {} will cause an error
        if "response_schema" in kwargs_dict:
            kwargs_dict["response_schema"] = self.adapt_schema(
                kwargs_dict["response_schema"]
            )

        return kwargs_dict

    @staticmethod
    def adapt_schema(schema):
        if type(schema) == str:
            schema = json.loads(schema)
        required_fields = schema.get("required", [])
        # drop the title, $schema, and $id and id fields unless they are required
        if "id" not in required_fields:
            schema.pop("id", None)
        if "title" not in required_fields:
            schema.pop("title", None)
        if "$schema" not in required_fields:
            schema.pop("$schema", None)
        if "$id" not in required_fields:
            schema.pop("$id", None)

        assert (
            is_json_schema_valid(schema) == True
        ), "The JSON schema after adaptation is no longer valid."
        return schema

    @property
    def max_ctx_len(self):
        return self.cfg.get("max_ctx_len", None)

    def count_tokens(self, text: str) -> int:
        response = self.model.count_tokens(text)
        return response.total_tokens
