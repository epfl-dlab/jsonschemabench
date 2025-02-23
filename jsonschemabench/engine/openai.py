import logging
import os
import time
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf
from openai._streaming import Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from ..utils import (
    Conversation,
    Token,
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
from ..config import OpenaiGenConfig
from validation import is_json_schema_valid


logger = logging.getLogger(__name__)


import os

try:
    from tiktoken import encoding_for_model
    from openai import OpenAI
except ImportError:
    print("openai is not installed, please install it to use openai models")
from typing import List, Optional


class OpenAIEngine(BaseEngine):

    config_cls = OpenaiGenConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.tokenizer = encoding_for_model(self.cfg.model_name)

    @staticmethod
    def init(cfg):
        return OpenAIEngine(cfg)

    def _generate_free_mode(
        self,
        prompt: str,
        generation_config: OpenaiGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        return self._call_platform(prompt, generation_config, metadata)

    def _generate_json_mode(
        self,
        prompt: str,
        generation_config: OpenaiGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:
        return self._call_platform(prompt, generation_config, metadata)

    def _generate_json_schema_mode(
        self,
        prompt: str,
        generation_config: OpenaiGenConfig,
        metadata: GenerationMetadata,
    ) -> Optional[str]:

        return self._call_platform(prompt, generation_config, metadata)

    @staticmethod
    def prepare_json_schema(kwargs_conf):
        pass

    def _call_platform(
        self,
        prompt: str,
        generation_config: OpenaiGenConfig,
        metadata: GenerationMetadata,
    ):

        system_msg_content = generation_config.get("system_message", None)
        system_message_obj = (
            {"role": "system", "content": system_msg_content}
            if system_msg_content
            else None
        )

        # currently only one prompt is supported, but more than one prompt can build a real conversation history
        conversation = Conversation(
            system_message=system_message_obj,
            user_messages=[{"role": "user", "content": prompt}],
        )

        kwargs_dict: dict = self.prepare_kwargs(generation_config.get("kwargs"))
        stream: bool = kwargs_dict.pop("stream", True)

        try:
            openai_completion = self.client.chat.completions.create(
                model=self.cfg.model_name,
                messages=conversation.to_messages(),
                stream=stream,
                logprobs=True,
                top_logprobs=20,  # max value allowed by OpenAI
                stream_options={"include_usage": True} if stream else None,
                **kwargs_dict,
            )
            if stream:
                response = self.get_openai_stream_response(openai_completion, prompt)
            else:
                response = self.get_openai_response(openai_completion, prompt)
        except Exception as e:
            error_message = str(e)
            logger.warn(f"Error message: {error_message}")
            metadata.compile_status = CompileStatus(
                code=CompileStatusCode.UNSUPPORTED_SCHEMA, message=str(e)
            )
            response = GenerationResponse(input=prompt, output="", metadata=metadata)
        response.input = prompt
        return response

    @profile_generation
    def generate(
        self,
        prompt: str,
        generation_config: OpenaiGenConfig,
        schema: Optional[str] = None,
    ) -> GenerationResponse:
        if schema is not None:
            generation_config = generation_config.update_json_schema(schema)
        """
        A list of messages comprising the conversation so far
        """
        gen_metadata = GenerationMetadata()
        gen_config: DictConfig = generation_config.gen_config

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

        response.set_stop_reason(gen_config.get("kwargs").get("max_tokens", None))

        return response

    def prepare_kwargs(self, kwargs_conf: dict) -> dict:
        """
        Prepare the kwargs for the OpenAI API call.
        """
        # convert kwargs to a dictionary if it is still OmegaConf
        if isinstance(kwargs_conf, DictConfig):
            kwargs_dict = OmegaConf.to_container(kwargs_conf)
        else:
            assert (
                type(kwargs_conf) == dict
            ), f"kwargs_conf should be a dictionary if not an OmegaConf object, but got {type(kwargs_conf)}"
            kwargs_dict = kwargs_conf if kwargs_conf else {}

        # prepare json_schema
        try:
            kwargs_dict["response_format"]["json_schema"]["schema"] = self.adapt_schema(
                kwargs_dict["response_format"]["json_schema"]["schema"]
            )
        except KeyError:
            pass

        return kwargs_dict

    def get_num_tokens(self, texts: List[str]) -> List[List[int]]:
        return self.tokenizer(texts, return_tensors=None)["input_ids"]

    @staticmethod
    def get_token_usage(last_chunk: ChatCompletionChunk) -> TokenUsage:
        usage = TokenUsage(
            input_tokens=last_chunk.usage.prompt_tokens,
            output_tokens=last_chunk.usage.completion_tokens,
        )
        return usage

    def update_total_usage(self, openai_response: GenerationResponse) -> None:
        self.total_usage["num_samples"] += 1
        self.total_usage["input_tokens"] += openai_response.token_usage.input_tokens
        self.total_usage["output_tokens"] += openai_response.token_usage.output_tokens

    def get_openai_stream_response(
        self, openai_generator: Stream[ChatCompletionChunk], prompt: str
    ) -> GenerationResponse:

        tokens_str: List[str] = []
        logprobs: List[float] = []
        top_tokens: List[List[Token]] = []

        for i, chunk in enumerate(openai_generator):
            if i == 0:
                first_token_arrival_time: float = time.time()

            if len(chunk.choices) == 0 or chunk.choices[0].finish_reason is not None:
                continue

            chunk_content = chunk.choices[0].delta.content
            if chunk_content == "":
                continue

            tokens_str.append(chunk_content)
            logprobs.append(chunk.choices[0].logprobs.content[0].logprob)

            new_top_tokens = []
            for token in chunk.choices[0].logprobs.content[0].top_logprobs:
                new_top_tokens.append(
                    Token(
                        id=self.convert_token_to_id(token.token),
                        text=token.token,
                        logprob=token.logprob,
                    )
                )
            top_tokens.append(new_top_tokens)

        # the usage is only available for the last chunk
        usage: TokenUsage = self.get_token_usage(chunk)

        metadata = GenerationMetadata(
            system_fingerprint=chunk.system_fingerprint,
            _first_tok_arr_time=first_token_arrival_time,
            compile_status=CompileStatus(code=CompileStatusCode.OK),
            decoding_status=DecodingStatus(code=DecodingStatusCode.OK),
        )

        tokens_ids = [self.convert_token_to_id(token) for token in tokens_str]

        response = GenerationResponse(
            input=prompt,
            output="".join(tokens_str),
            generated_tokens=[
                Token(id=id, text=token, logprob=logprob)
                for id, token, logprob in zip(tokens_ids, tokens_str, logprobs)
            ],
            top_tokens=top_tokens,
            metadata=metadata,
            # compile_status=None,
            perf_metrics=None,
            token_usage=usage,
        )

        return response

    def get_openai_response(
        self, openai_response: ChatCompletion, prompt: str
    ) -> GenerationResponse:
        output = openai_response.choices[0].message.content

        tokens_str = [
            token.token for token in openai_response.choices[0].logprobs.content
        ]

        logprobs = [
            token.logprob for token in openai_response.choices[0].logprobs.content
        ]

        top_tokens = []
        for token in openai_response.choices[0].logprobs.content:
            new_top_tokens = []
            for top_logprob in token.top_logprobs:
                new_top_tokens.append(
                    Token(
                        id=self.convert_token_to_id(top_logprob.token),
                        text=top_logprob.token,
                        logprob=top_logprob.logprob,
                    )
                )
            top_tokens.append(new_top_tokens)

        usage: TokenUsage = TokenUsage(
            input_tokens=openai_response.usage.prompt_tokens,
            output_tokens=openai_response.usage.completion_tokens,
        )

        metadata = GenerationMetadata(
            system_fingerprint=openai_response.system_fingerprint,
            compile_status=CompileStatus(code=CompileStatusCode.OK),
            decoding_status=DecodingStatus(code=DecodingStatusCode.OK),
        )

        tokens_ids = [self.convert_token_to_id(token) for token in tokens_str]

        response = GenerationResponse(
            input=prompt,
            output=output,
            generated_tokens=[
                Token(id=id, text=token, logprob=logprob)
                for id, token, logprob in zip(tokens_ids, tokens_str, logprobs)
            ],
            top_tokens=top_tokens,
            metadata=metadata,
            compile_status=None,
            perf_metrics=None,
            token_usage=usage,
        )

        return response

    @staticmethod
    def adapt_schema(schema: dict):
        """
        Preprocess a JSON schema for OpenAI.

        :param schema: A JSON schema.
        :return: A preprocessed JSON schema.
        """
        # Remove unsupported keywords
        if type(schema) == str:
            raise ValueError("The schema should be a dictionary, not a string.")
        recursively_set_additional_properties_false(schema)
        add_root_type_if_missing(schema)
        schema = set_all_properties_required(schema)
        if not is_json_schema_valid(schema):
            logger.warn("The JSON schema after adaptation is no longer valid.")
        return schema


    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def max_ctx_len(self):
        max_ctx_len_dict = {
            "gpt-4o": 128 * 1000,
            "gpt-4o-2024-08-06": 128 * 1000,
            "gpt-4o-2024-05-13": 128 * 1000,
            "gpt-4o-mini": 128 * 1000,
        }
        return max_ctx_len_dict[self.cfg.model_name]


def add_root_type_if_missing(schema: dict):
    if "type" not in schema:
        schema["type"] = "object"


def set_all_properties_required(schema: object) -> object:
    if not isinstance(schema, dict):
        return schema
    if "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
    for value in schema.values():
        if isinstance(value, dict):
            set_all_properties_required(value)
        elif isinstance(value, list):
            for item in value:
                set_all_properties_required(item)
    return schema

def recursively_set_additional_properties_false(schema: dict):
    if not isinstance(schema, dict):
        return
    if (
        "additionalProperties" not in schema or schema["additionalProperties"]
    ) and schema.get("properties"):
        schema["additionalProperties"] = False
    if "properties" in schema:
        for prop in schema["properties"]:
            recursively_set_additional_properties_false(schema["properties"][prop])
    if "items" in schema:
        recursively_set_additional_properties_false(schema["items"])


OPENAI_JSON_ERRORS = {
    "Expected at most 500 enum values in total within a single schema when using structured outputs, but received": "TooManyEnumValues",
    "is not permitted": "InvalidKeyword",
    "Unsupported keywords": "UnsupportedKeywords",
    "'required' is required to be supplied and to be an array including every key in properties": "RequiredPropertiesMissing",
    "array schema missing items": "ArrayItemsMissing",
    """schema must be a JSON Schema of \\'type: "object"\\'""": "InvalidType",
    "object schema missing properties.": "ObjectMissingProperties",
    "'additionalProperties' is required to be supplied and to be false": "AdditionalPropertiesRequired",
    "schema must have a 'type' key": "TypeKeyMissing",
    "Please ensure it is a valid JSON Schema.": "OversizeSchema",
    "$ref cannot have keywords": "InvalidRefUsage",
    "reference can only point to definitions defined at the top level of the schema": "NonTopLevelRef",
}
