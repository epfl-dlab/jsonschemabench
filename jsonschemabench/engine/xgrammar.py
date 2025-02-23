

import time
import torch
import xgrammar as xgr
from transformers.utils import ModelOutput
from transformers.generation import GenerationConfig
from transformers import LogitsProcessor
import logging
import time
from typing import List, Optional, Union
from omegaconf import DictConfig, OmegaConf

from ..utils import (
    COMPILATION_TIMEOUT,
    GENERATION_TIMEOUT,
    GenerationMetadata,
    GenerationResponse,
    CompileStatus,
    CompileStatusCode,
    DecodingStatus,
    DecodingStatusCode,
    profile_generation,
)
from ._engine import HFModel
from ..config import XGrammarGenConfig
import stopit
from json import dumps
from omegaconf import DictConfig, OmegaConf
from typing import Dict, List, Optional, Union


logger = logging.getLogger(__name__)


class TimingLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()
        self.timestamps = []

    def __call__(self, input_ids, scores):
        """
        Called during decoding to process the logits.

        Args:
            input_ids (torch.Tensor): The input token IDs.
            scores (torch.Tensor): The logits scores for the next token.

        Returns:
            torch.Tensor: The unchanged scores.
        """
        # Record the current timestamp
        current_time = time.time()
        self.timestamps.append(current_time)

        # Return the logits unchanged
        return scores


def try_compile_json(
    json_schema_obj: Union[dict, str],
    grammar_compiler: xgr.GrammarCompiler,
    timeout: int = COMPILATION_TIMEOUT,
) -> xgr.compiler.CompiledGrammar:
    json_schema_str = (
        json_schema_obj if type(json_schema_obj) == str else dumps(json_schema_obj)
    )
    with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
            compiled_grammar = grammar_compiler.compile_json_schema(json_schema_str)

    # Check the state of the timeout context manager after the block
    if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
        return compiled_grammar
    elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
        raise TimeoutError("Compilation timed out")
    else:
        raise RuntimeError("Unexpected error during JSON compilation")


class XGrammarEngine(HFModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            self.hf_tokenizer, vocab_size=self.hf_model.config.vocab_size
        )
        self.grammar_compiler = xgr.GrammarCompiler(
            tokenizer_info, cache_enabled=self.cfg.get("grammar_cache_enabled", False)
        )

    @staticmethod
    def init(cfg):
        return XGrammarEngine(cfg)

    def call_engine(
        self,
        model_input: Dict[str, torch.Tensor],
        generation_config: DictConfig,
        stream=True,
        metadata: GenerationMetadata = None,
        timeout=GENERATION_TIMEOUT,
        logits_processor: Optional[LogitsProcessor] = None,
    ) -> Optional[ModelOutput]:
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            if to_ctx_mgr.state == to_ctx_mgr.EXECUTING:
                try:
                    model_output: ModelOutput = self.hf_model.generate(
                        model_input["input_ids"],
                        generation_config=GenerationConfig(
                            **generation_config.gen_config.kwargs
                        ),
                        attention_mask=model_input["attention_mask"],
                        tokenizer=self.hf_tokenizer,
                        logits_processor=logits_processor,
                    )
                    metadata.decoding_status = DecodingStatus(
                        code=DecodingStatusCode.OK
                    )
                except BaseException as e:
                    logger.warn(f"Error generating text: {str(e)}")
                    metadata.decoding_status = DecodingStatus(
                        code=DecodingStatusCode.UNKOWN_ERROR, message=str(e)
                    )
                    # model_output = ModelOutput(sequences=torch.empty((0, model_input["input_ids"].shape[1]), dtype=torch.int64))
                    model_output = None
        # Check the state of the timeout context manager after the block
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return model_output
        elif to_ctx_mgr.state == to_ctx_mgr.TIMED_OUT:
            raise TimeoutError("Generation timed out")
        else:
            raise RuntimeError("Unexpected error during generation")

    @profile_generation
    def generate(
        self,
        prompt: str,
        generation_config: XGrammarGenConfig,
        schema: Optional[str] = None,
    ) -> GenerationResponse:
        # streaming mode is missing yet https://github.com/flozi00/atra/blob/375bd740c37fb42d35048ae33ae414841f22938a/atra/text_utils/chat.py#LL98C7-L98C7
        # maybe a simpler way is to create a logit processor(metadata) to get the ttft

        if schema is not None:
            generation_config = generation_config.update_json_schema(schema)

        timing_logits_processor = TimingLogitsProcessor()
        logits_processor = [timing_logits_processor]
        metadata = GenerationMetadata()

        if generation_config.gen_config.mode == "json_schema":
            schema: Union[dict, str] = generation_config.gen_config.json_schema

            if schema is not None:
                compiled_grammar = self.compile_grammar(schema, metadata)

                if compiled_grammar is not None:
                    logits_processor.append(
                        xgr.contrib.hf.LogitsProcessor(compiled_grammar)
                    )
        elif generation_config.gen_config.mode == "json_mode":
            compiled_grammar = self.grammar_compiler.compile_builtin_json_grammar()
            logits_processor.append(xgr.contrib.hf.LogitsProcessor(compiled_grammar))

        # Generate text from the model
        model_input: Dict[str, torch.Tensor] = self.hf_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
        ).to("cuda")

        # if the compilation is successful or no compilation is needed, generate text
        if metadata.compile_status.code in (
            CompileStatusCode.OK,
            CompileStatusCode.TBD,
        ):

            model_output = self.call_engine(
                model_input=model_input,
                generation_config=generation_config,
                stream=True,
                metadata=metadata,
                timeout=GENERATION_TIMEOUT,
                logits_processor=logits_processor,
            )
            if model_output is None:
                output_text = None
            else:
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

                metadata.decoding_status = DecodingStatus(code=DecodingStatusCode.OK)
                metadata._first_tok_arr_time = timing_logits_processor.timestamps[0]
                output_text: str = best_texts[0]  # currently we only support bs=1
        else:
            model_output = None
            output_text = None

        token_usage = self.get_token_usage(model_input, model_output)

        gen_response = GenerationResponse(
            input=prompt,
            output=output_text,
            # generated_tokens=[
            #     Token(id=self.convert_token_to_id(token_str), text=token_str)
            #     for token_str in output_texts
            # ],
            metadata=metadata,
            token_usage=token_usage,
        )
        return gen_response

    def compile_grammar(
        self, schema: Union[dict, str], metadata: GenerationMetadata
    ) -> Optional[xgr.compiler.CompiledGrammar]:
        if isinstance(schema, DictConfig):
            schema = OmegaConf.to_container(schema)
        compile_status = metadata.compile_status
        json_schema_str = schema if type(schema) == str else dumps(schema)

        compiled_grammar = None
        try:
            compiled_grammar = try_compile_json(
                json_schema_str, self.grammar_compiler, COMPILATION_TIMEOUT
            )
            compile_status = CompileStatus(code=CompileStatusCode.OK)
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
        metadata._grammar_compilation_end_time = time.time()

        return compiled_grammar
