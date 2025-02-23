import logging
import os
import time
import json
from typing import Iterator, List, Optional, Union
import numpy as np
from omegaconf import DictConfig, OmegaConf
import stopit
from ..utils import (
    COMPILATION_TIMEOUT,
    Conversation,
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
from ..config import LlamaCppGenConfig

from llama_cpp import Llama
import llama_cpp
from llama_cpp._internals import LlamaSampler, LlamaModel
from llama_cpp.llama_grammar import LlamaGrammar, JSON_GBNF

from llama_cpp.llama_types import (
    CreateChatCompletionStreamResponse,
    CreateChatCompletionResponse,
)

JSON_MODE_GBNF = LlamaGrammar.from_string(JSON_GBNF, verbose=False)


logger = logging.getLogger(__name__)

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())


def try_compile_json(
    json_schema_obj: Union[dict, str], timeout: int = COMPILATION_TIMEOUT
) -> LlamaGrammar:
    json_schema_str = (
        json_schema_obj if type(json_schema_obj) == str else json.dumps(json_schema_obj)
    )
    with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
            grammar = LlamaGrammar.from_json_schema(json_schema_str, verbose=False)

    # Check the state of the timeout context manager after the block
    if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
        return grammar
    elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
        raise TimeoutError("Compilation timed out")
    else:
        raise RuntimeError("Unexpected error during JSON compilation")


def try_sampler_segfault(model: LlamaModel, grammar: LlamaGrammar) -> dict:
    """
    llama-cpp could have segmentation fault when adding grammar to sampler
    """

    def child_process():
        import signal

        signal.signal(signal.SIGALRM, lambda signum, frame: os._exit(2))
        signal.alarm(15)
        try:
            LlamaSampler().add_grammar(model, grammar)
            os._exit(0)
        except Exception:
            os._exit(1)

    id = os.fork()
    if id == 0:
        child_process()
    else:
        _, status = os.waitpid(id, 0)
        if os.WIFEXITED(status):
            exit_code = os.WEXITSTATUS(status)
            return {"success": exit_code == 0, "exit_code": exit_code}
        return {"success": False, "error": "Unknown status"}


class LlamaCppEngine(BaseEngine):

    config_cls = LlamaCppGenConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        self.unmasked_logprobs: List[np.array] = []
        # logits_all is required to return the logprobs
        cfg.kwargs["logits_all"] = True
        self.model = Llama.from_pretrained(cfg.repo_id, cfg.filename, **cfg.kwargs)

    @staticmethod
    def init(cfg):
        return LlamaCppEngine(cfg)

    def clear_kv_cache(self):
        self.model.reset()

    def get_kv_cache_token_count(self):
        return llama_cpp.llama_get_kv_cache_token_count(self.model._ctx.ctx)

    def _generate_free_mode(
        self,
        prompt: str,
        generation_config: LlamaCppGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        grammar = None

        response = self._call_engine(prompt, generation_config, grammar, metadata)

        return response

    def _generate_json_mode(
        self,
        prompt: str,
        generation_config: LlamaCppGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        grammar = JSON_MODE_GBNF

        response = self._call_engine(prompt, generation_config, grammar, metadata)

        return response

    def _generate_json_schema_mode(
        self,
        prompt: str,
        generation_config: LlamaCppGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:

        # schema = kwargs_dict["response_format"].get("schema", None)
        schema: Union[dict, str] = generation_config.kwargs.get("response_format").get(
            "schema", None
        )

        # kwargs_dict.pop("response_format", None)

        grammar = self.compile_grammar(schema, metadata)

        response = self._call_engine(prompt, generation_config, grammar, metadata)

        return response

    @staticmethod
    def prepare_json_schema(kwargs_conf):
        pass

    def _call_engine(
        self,
        prompt: str,
        generation_config: LlamaCppGenConfig,
        grammar: LlamaGrammar = None,
        metadata: GenerationMetadata = None,
    ) -> Optional[str]:
        kwargs_dict: dict = OmegaConf.to_container(generation_config.kwargs)
        stream: bool = generation_config.get("stream", False)

        def save_unmasked_logprobs(_, logits):
            self.unmasked_logprobs.append(log_softmax(np.array(logits)))
            return logits

        conversation = Conversation(
            system_message=generation_config.get("system_message", None),
            user_messages=[{"role": "user", "content": prompt}],
        )
        if metadata.is_valid_so_far():
            generator = self.model.create_chat_completion(
                messages=conversation.to_messages(),
                stream=stream,
                logprobs=True,
                # top_logprobs=20,  # to match openai TODO, this parameter slows down the generation by 200%
                # logits_processor=[save_unmasked_logprobs],
                grammar=grammar,
                **kwargs_dict,
            )
            if stream:
                response = self.get_stream_response(generator, prompt, metadata)
            else:
                response = self.get_response(generator, prompt, metadata)
        else:
            response = GenerationResponse(input=prompt, output="", metadata=metadata)

        return response

    @profile_generation
    def generate(
        self,
        prompt: str,
        generation_config: LlamaCppGenConfig,
        schema: Optional[str] = None,
    ) -> GenerationResponse:
        if schema is not None:
            generation_config = generation_config.update_json_schema(schema)
        self.unmasked_logprobs = []
        gen_metadata = GenerationMetadata()

        gen_config = generation_config.gen_config

        if gen_config.mode == "json_schema":
            response = self._generate_json_schema_mode(prompt, gen_config, gen_metadata)
        elif gen_config.mode == "json_mode":
            response = self._generate_json_mode(prompt, gen_config, gen_metadata)
        elif gen_config.mode == "free":
            response = self._generate_free_mode(prompt, gen_config, gen_metadata)
        else:
            raise NotImplementedError

        response.set_stop_reason(gen_config.get("kwargs").get("max_tokens", None))

        return response

    def get_stream_response(
        self,
        generator: Iterator[CreateChatCompletionStreamResponse],
        prompt: str,
        metadata: GenerationMetadata,
    ) -> GenerationResponse:

        tokens_str: List[str] = []

        for i, chunk in enumerate(generator):
            if i == 0:
                first_token_arrival_time: float = time.time()

            if (
                len(chunk["choices"]) == 0
                or chunk["choices"][0]["finish_reason"] is not None
            ):
                continue

            chunk_content = chunk["choices"][0]["delta"].get("content", "")
            if chunk_content == "":
                continue

            tokens_str.append(chunk_content)

        generation = "".join(tokens_str)
        usage: TokenUsage = self.get_token_usage(prompt, generation)

        # the usage is only available for the last chunk
        metadata._first_tok_arr_time = first_token_arrival_time

        metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)

        response = GenerationResponse(
            input=prompt,
            output=generation,
            metadata=metadata,
            perf_metrics=None,
            token_usage=usage,
        )
        return response

    def get_response(
        self,
        response: CreateChatCompletionResponse,
        prompt: str,
        metadata: GenerationMetadata,
    ) -> GenerationResponse:
        output = response["choices"][0]["message"]["content"]

        usage: TokenUsage = TokenUsage(
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
        )

        metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)

        response = GenerationResponse(
            input=prompt,
            output=output,
            metadata=metadata,
            token_usage=usage,
        )
        return response

    def compile_grammar(
        self, schema: Union[dict, str], metadata: GenerationMetadata
    ) -> LlamaGrammar:
        grammar = None
        if isinstance(schema, DictConfig):
            schema = OmegaConf.to_container(schema)
        compile_status = metadata.compile_status
        try:
            # grammar = LlamaGrammar.from_json_schema(dumps(schema), verbose=False)
            grammar = try_compile_json(schema)

            segfault_check_result = try_sampler_segfault(self.model._model, grammar)
            if segfault_check_result["success"]:
                metadata._grammar_compilation_end_time = time.time()
                compile_status = CompileStatus(code=CompileStatusCode.OK)
            else:
                logger.warn(
                    f"Failed to add grammar to sampler: {segfault_check_result}"
                )
                compile_status = CompileStatus(
                    code=CompileStatusCode.UNSUPPORTED_SCHEMA,
                    message=f"Failed to add grammar to sampler",
                )
        except TimeoutError as e:
            logger.warn(f"Error compiling grammar: {str(e)}")
            compile_status = CompileStatus(
                code=CompileStatusCode.COMPILE_TIMEOUT, message=str(e)
            )
        except Exception as e:
            logger.warn(f"Error compiling grammar: {str(e)}")
            compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
        metadata.compile_status = compile_status

        return grammar

    def get_token_usage_non_streaming(
        self, llama_cpp_response: Optional[dict]
    ) -> dict[str, int]:
        if llama_cpp_response is None:
            return {"input_tokens": 0, "output_tokens": 0}
        usage = {
            "input_tokens": llama_cpp_response["usage"]["prompt_tokens"],
            "output_tokens": llama_cpp_response["usage"]["completion_tokens"],
        }
        return usage

    def get_token_usage(self, input: str, output: str) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.count_tokens(input),
            output_tokens=self.count_tokens(output),
        )

    @property
    def max_ctx_len(self):
        return self.model.n_ctx()

    def encode(self, text: str) -> List[int]:
        byte_string = text.encode("utf-8")
        return self.model.tokenize(byte_string)

    def decode(self, ids: List[int]) -> str:
        byte_string = self.model.detokenize(ids)
        return byte_string.decode("utf-8")
