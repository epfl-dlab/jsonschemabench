import logging
import time
import json
from typing import List, Optional
from omegaconf import DictConfig
import omegaconf
import outlines
from ..utils import (
    COMPILATION_TIMEOUT,
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
from ..config import OutlinesGenConfig

from outlines.caching import cache_disabled
import json
import stopit


logger = logging.getLogger(__name__)

def try_compile_json(
    model, json_schema_obj, timeout=COMPILATION_TIMEOUT
) -> outlines.generate.api.SequenceGeneratorAdapter:
    with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
            result = outlines.generate.json(
                model, schema_object=json.dumps(json_schema_obj)
            )

    # Check the state of the timeout context manager after the block
    if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
        return result
    elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
        raise TimeoutError("Compilation timed out")
    else:
        raise RuntimeError("Unexpected error during JSON compilation")


class OutlinesEngine(BaseEngine):

    config_cls = OutlinesGenConfig

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
                self.model = outlines.models.transformers(
                    cfg.model_name,
                    model_kwargs={
                        "torch_dtype": torch.bfloat16,
                        "quantization_config": bnb_config,
                        "device_map": "auto",
                    },
                )
            else:
                self.model = outlines.models.transformers(
                    cfg.model_name,
                    model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
                )
                self.tokenizer = self.model.tokenizer
        elif cfg.engine == "llamacpp":
            import llama_cpp

            # https://github.com/dottxt-ai/outlines/issues/1261
            # manually setting the tokenizer to avoid tokenizer issue which is the case for llama3.1
            if cfg.get("hf_tokenizer_id", None):
                tokenizer = llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
                    cfg.get("hf_tokenizer_id")
                )
            else:
                tokenizer = None
            self.model = outlines.models.llamacpp(
                repo_id=cfg.repo_id,
                filename=cfg.filename,
                tokenizer=tokenizer,
                **cfg.kwargs,
            )
            self._max_ctx_len = cfg.kwargs.get("n_ctx", 1024)

    @staticmethod
    def init(cfg):
        return OutlinesEngine(cfg)

    # def count_tokens(self, text: str) -> int:
    #     return len(self.model.model.tokenizer().encode(text))

    def encode(self, text: str) -> List[int]:
        if self.cfg.engine == "hf":
            return self.model.tokenizer.encode(text)[0].tolist()[0]
        elif self.cfg.engine == "llamacpp":
            return self.model.model.tokenizer().encode(text)
        else:
            raise NotImplementedError

    ### WARNING: This works only for llama_cpp => for transformers, it doesn't work
    def decode(self, ids: List[int]) -> str:
        return self.model.model.tokenizer().decode(ids)

    ### END WARNING ###

    def clear_kv_cache(self):
        if self.cfg.engine == "llamacpp":
            self.model.model.reset()
        else:
            raise NotImplementedError

    def get_kv_cache_token_count(self):
        if self.cfg.engine == "llamacpp":
            import llama_cpp

            llama_cpp.llama_get_kv_cache_token_count(self.model.model._ctx.ctx)
        else:
            raise NotImplementedError

    @property
    def max_ctx_len(self):
        return self._max_ctx_len

    def preprocess_prompts_before_run(self, prompts, generation_config):
        model_inputs = prompts
        return model_inputs

    def compile_grammar(self, json_schema_obj, metadata: GenerationMetadata):
        generator = None
        if isinstance(json_schema_obj, omegaconf.dictconfig.DictConfig):
            json_schema_obj = omegaconf.OmegaConf.to_container(json_schema_obj)

        try:
            generator: outlines.generate.api.SequenceGeneratorAdapter = (
                try_compile_json(self.model, json_schema_obj)
            )
            compile_status = CompileStatus(code=CompileStatusCode.OK)
        except TimeoutError as e:
            logger.warn(f"Error message: {str(e)}")
            compile_status = CompileStatus(
                code=CompileStatusCode.COMPILE_TIMEOUT, message=str(e)
            )
        except BaseException as e:
            # the reason whe we use BaseException instead of Exception is because we need to handle pyo3_runtime.PanicException which is a subclass of BaseException
            # it is yielded when the underlying Rust code panics
            logger.warn(f"Error message: {str(e)}")
            compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
        metadata._grammar_compilation_end_time = time.time()
        metadata.compile_status = compile_status
        return generator

    def _generate_free_mode(
        self,
        prompt: str,
        generation_config: OutlinesGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        generator = outlines.generate.text(
            self.model,
        )
        stream: bool = generation_config.get("stream", True)
        output = self.call_engine(
            generator, prompt, generation_config, meta_data=metadata, stream=stream
        )
        return output

    def _generate_json_mode(
        self,
        prompt: str,
        generation_config: OutlinesGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        from outlines.grammars import json as json_grammar

        stream: bool = generation_config.get("stream", True)
        generator = outlines.generate.cfg(self.model, json_grammar)
        output = self.call_engine(
            generator, prompt, generation_config, meta_data=metadata, stream=stream
        )

    def _generate_json_schema_mode(
        self,
        prompt: str,
        generation_config: OutlinesGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        stream: bool = generation_config.get("stream", True)
        json_schema_obj = generation_config.json_schema

        if self.cfg.get("grammar_cache_enabled", False):
            generator = self.compile_grammar(json_schema_obj, metadata)
        else:
            with cache_disabled():
                generator = self.compile_grammar(json_schema_obj, metadata)

        output = None
        # if the grammar is compiled successfully, we can decode
        if metadata.compile_status.code == CompileStatusCode.OK:
            compile_status = metadata.compile_status
            decoding_status = metadata.decoding_status
            try:
                output = self.call_engine(
                    generator,
                    prompt,
                    generation_config,
                    meta_data=metadata,
                    stream=stream,
                )
                decoding_status = DecodingStatus(code=DecodingStatusCode.OK)
            except TimeoutError as e:
                logger.warn(f"Error message: {str(e)}")
                decoding_status = DecodingStatus(
                    code=DecodingStatusCode.DECODING_TIMEOUT, message=str(e)
                )
            except Exception as e:
                logger.warn(f"Error message: {str(e)}")
                decoding_status = DecodingStatus(
                    code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
                )
            metadata.compile_status = compile_status
            metadata.decoding_status = decoding_status
        else:
            pass
        return output

    @staticmethod
    def prepare_json_schema(kwargs_conf):
        pass

    @profile_generation
    def generate(
        self,
        prompt: str,
        generation_config: OutlinesGenConfig,
        schema: Optional[str] = None,
    ) -> GenerationResponse:
        if schema is not None:
            generation_config = generation_config.update_json_schema(schema)

        gen_metadata = GenerationMetadata()

        gen_config = generation_config.gen_config

        if gen_config.mode == "json_schema":
            generation: Optional[str] = self._generate_json_schema_mode(
                prompt, gen_config, gen_metadata
            )

        elif gen_config.mode == "json_mode":
            generation: Optional[str] = self._generate_json_mode(
                prompt, gen_config, gen_metadata
            )
        elif gen_config.mode == "free":
            generation: Optional[str] = self._generate_free_mode(
                prompt, gen_config, gen_metadata
            )
        else:
            raise NotImplementedError

        # in case of json schema, output from Outlines is a dict
        if isinstance(generation, dict):
            generation = json.dumps(generation)
        # sometimes the outlines failed to generate a valid output and will return an empty list as output
        elif generation is None:
            generation = ""
        elif type(generation) != str:
            generation = str(generation)
        usage = self.get_usage(prompt, generation)

        response = GenerationResponse(
            output=generation,
            input=prompt,
            token_usage=usage,
            metadata=gen_metadata,
            perf_metrics=None,
        )

        response.set_stop_reason(gen_config.get("gen_kwargs").get("max_tokens", None))
        return response

    def get_usage(self, prompt: str, output: str) -> TokenUsage:
        # usage = {
        #     "input_tokens": self._estimate_token_count(prompt),
        #     "output_tokens": self._estimate_token_count(output),
        # }
        usage = TokenUsage(
            input_tokens=self._estimate_token_count(prompt),
            output_tokens=self._estimate_token_count(output),
        )
        return usage

    @staticmethod
    def call_engine(
        generator: outlines.generate.api.SequenceGeneratorAdapter,
        prompt: str,
        generation_config: DictConfig,
        stream=True,
        meta_data: GenerationMetadata = None,
        timeout=GENERATION_TIMEOUT,
    ):
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                # Attempt to generate the output
                if stream:
                    token_iterator = generator.stream(
                        prompt, **generation_config.gen_kwargs
                    )
                    tokens: List[str] = []
                    for i, token in enumerate(token_iterator):
                        if i == 0 and meta_data is not None:
                            meta_data._first_tok_arr_time = time.time()
                        tokens.append(token)
                    output = "".join(tokens)
                else:
                    output = generator(prompt, **generation_config.gen_kwargs)

        # Check the state of the timeout context manager after the block
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            meta_data.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)
            return output
        elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            raise TimeoutError("Generation timed out")
        else:
            raise RuntimeError("Unexpected error during generation")

    def _estimate_token_count(self, text: str) -> int:
        # in some cases, the outlines failed to generate a valid output and will return an empty list
        if type(text) != str:
            text = str(text)
        if self.cfg.engine == "hf":
            input_ids, att_mask = self.model.tokenizer.encode(text)
        elif self.cfg.engine == "llamacpp":
            input_ids = self.model.model.tokenizer().encode(text)
        return len(input_ids)
