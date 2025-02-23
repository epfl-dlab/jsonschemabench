from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import math
import time
from typing import Callable, Dict, Optional, List

from .config import GenConfig

COMPILATION_TIMEOUT = 40  # 10 in previous version
GENERATION_TIMEOUT = 60  # 20 in previous version

class CompileStatusCode(Enum):
    TBD = -1
    OK = 0
    UNSUPPORTED_SCHEMA = 1
    RUNTIME_GRAMMAR_ERROR = 2
    API_BAD_RESPONSE = 3
    PROMPT_TOO_LONG = 4
    COMPILE_TIMEOUT = 5
    RUNTIME_TIMEOUT = 6
    UNKOWN_ERROR = 7


class DecodingStatusCode(Enum):
    TBD = -1
    OK = 0
    EXCEEDING_MAX_CTX = 1
    DECODING_TIMEOUT = 2
    BAD_API_RESPONSE = 3
    UNKOWN_ERROR = 4


class JsonSchemaMatchingCode(Enum):
    TBD = -1
    MATCH = 0
    SYNTAX_ERROR = 1
    SEMANTIC_ERROR = 2
    JSON_NOT_FOUND_ERROR = 3
    # INVALID_REF_JSON_SCHEMA_ERROR = -1
    UNKOWN_ERROR = 4
    SKIPPED = 5
    EMPTY_INPUT_OR_BAD_FORMAT = 6


class ExactMatchStatusCode(Enum):
    TBD = -1
    MATCH = 0
    MISMATCH = 1
    EMPTY_INPUT_OR_BAD_FORMAT = 2


@dataclass
class CompileStatus:
    code: CompileStatusCode = CompileStatusCode.TBD
    message: str = "unknown"


@dataclass
class DecodingStatus:
    code: DecodingStatusCode = DecodingStatusCode.TBD
    message: str = "unknown"


# beta , WIP
@dataclass
class JSMatchStatus:
    code: JsonSchemaMatchingCode = JsonSchemaMatchingCode.TBD
    message: str = "unknown"


# beta , WIP
@dataclass
class EvalMatchStatus:
    code: ExactMatchStatusCode = ExactMatchStatusCode.TBD
    message: str = "unknown"


# beta , WIP
@dataclass
class EvaluationStatus:
    js_match_status: JSMatchStatus = JSMatchStatus()
    eval_match_status: EvalMatchStatus = EvalMatchStatus()


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    no_ff_output_tokens: int = 0  # legacy field, should not be used anymore
    ff_output_tokens: int = 0


@dataclass
class TokenDiff:
    index: int
    original_id: int
    original_token: str
    encoded_id: int
    encoded_token: str


@dataclass
class TokenizationAnalysis:
    match: bool = False
    original_length: int = 0
    encoded_length: int = 0
    num_differing_tokens: int = 0
    first_divergence_index: Optional[int] = None
    differing_token_indices: List[int] = field(default_factory=list)
    token_diffs: List[TokenDiff] = field(default_factory=list)


@dataclass
class Token:
    id: int
    text: Optional[str] = None
    logprob: Optional[float] = None
    unmasked_logprob: Optional[float] = None



@dataclass
class GenerationMetadata:
    system_fingerprint: Optional[str] = None
    _first_tok_arr_time: Optional[float] = None
    _grammar_compilation_end_time: Optional[float] = None
    compile_status: Optional[CompileStatus] = field(default_factory=CompileStatus)
    decoding_status: Optional[DecodingStatus] = field(default_factory=DecodingStatus)

    def is_valid_so_far(self):
        compile_ok = self.compile_status.code in [
            CompileStatusCode.OK,
            CompileStatusCode.TBD,
        ]
        decode_ok = self.decoding_status.code in [
            DecodingStatusCode.OK,
            DecodingStatusCode.TBD,
        ]

        return compile_ok and decode_ok


@dataclass
class PerfMetrics:

    ttft: float = 0.0  # Time to first token in s
    tpot: float = 0.0  # Time per output token in ms
    tgt: float = 0.0  # Total generation time in s
    gct: float = 0.0  # Grammar compilation time in s
    prft: float = 0.0  #  prefilling time in s
    peak_memory: float = 0.0  # in MB

    @classmethod
    def from_timestamps(
        cls,
        start_time: float,
        grammar_compilation_end_time: Optional[float],
        first_token_arrival_time: Optional[float],
        end_time: float,
        num_output_tokens: int,
    ):
        ttft = safe_subtract(first_token_arrival_time, start_time)
        tpot = safe_divide(
            safe_subtract(end_time, first_token_arrival_time),
            safe_subtract(num_output_tokens, 1),
        )
        tgt = safe_subtract(end_time, start_time)
        gct = safe_subtract(grammar_compilation_end_time, start_time)
        prft = safe_subtract(first_token_arrival_time, grammar_compilation_end_time)
        return cls(
            ttft=ttft,
            tpot=tpot * 1000 if tpot is not None else None,
            tgt=tgt,
            gct=gct,
            prft=prft,
        )


class Stat(Enum):
    MEAN = "mean"
    STD = "std"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P25 = "25%"
    P50 = "50%"
    P75 = "75%"


class Conversation:
    def __init__(
        self,
        system_message: Optional[Dict[str, str]] = None,
        user_messages: Optional[List[Dict[str, str]]] = None,
    ):
        self.system_message = system_message
        self.user_messages = user_messages or []

    def to_messages(self) -> List[Dict[str, str]]:
        messages = []
        if self.system_message:
            messages.append(self.system_message)
        if self.user_messages:
            messages.extend(self.user_messages)
        return messages

    def to_texts(self) -> List[str]:
        return [msg["content"] for msg in self.to_messages()]


class GenerationResponse:
    def __init__(
        self,
        input: str,
        output: str,
        id: str = None,
        json_schema: str = None,
        label: Optional[str] = None,
        generated_tokens: Optional[List[Token]] = None,
        top_tokens: Optional[List[List[Token]]] = None,
        metadata: Optional[GenerationMetadata] = None,
        token_usage: Optional[TokenUsage] = None,
        perf_metrics: Optional[PerfMetrics] = None,
        tokenization_analysis: Optional[TokenizationAnalysis] = None,
    ):
        self.id = id
        self.input = input
        self.output = output
        self.label = label
        self.json_schema = json_schema
        self.generated_tokens = generated_tokens
        self.top_tokens = top_tokens
        self.metadata: GenerationMetadata = metadata or GenerationMetadata()
        self.token_usage: TokenUsage = token_usage or TokenUsage()
        self.perf_metrics: PerfMetrics = perf_metrics or PerfMetrics()
        self.tokenization_analysis: TokenizationAnalysis = (
            tokenization_analysis or TokenizationAnalysis()
        )

    def set_label(self, label: str):
        self.label = label

    def set_id(self, id: str):
        self.id = id

    def set_json_schema(self, json_schema: str):
        self.json_schema = json_schema

    def set_stop_reason(self, max_tokens: int):
        if self.token_usage.output_tokens >= max_tokens:
            # self.stop_reason = StopReason.REACHED_MAX_TOKENS
            self.metadata.decoding_status = DecodingStatus(
                code=DecodingStatusCode.EXCEEDING_MAX_CTX,
            )
        else:
            pass

    def __str__(self):
        string = ""
        for key, value in self.__dict__.items():
            string += f"{key}: {value}\n"
        return string



def round_dict(data, sig_figs=3):
    if isinstance(data, dict):
        # If data is a dictionary, apply rounding to each value
        return {key: round_dict(value, sig_figs) for key, value in data.items()}
    elif isinstance(data, list):
        # If data is a list, apply rounding to each item
        return [round_dict(item, sig_figs) for item in data]
    elif isinstance(data, float):
        # If data is a float, round it
        return round_to_sig_figs(data, sig_figs)
    else:
        # For other data types, return as is
        return data


def round_to_sig_figs(number, sig_figs):
    """Round a number to a specified number of significant figures."""
    # Handle special cases
    if math.isnan(number):
        return number
    if number == 0:
        return 0

    return round(number, sig_figs - int(math.floor(math.log10(abs(number)))) - 1)


def safe_divide(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely divides a by b, returning None if either input is None."""
    if a is None or b is None or b == 0:
        return None
    return a / b


def safe_subtract(a: Optional[float], b: Optional[float]) -> Optional[float]:
    """Safely subtracts b from a, returning None if either input is None."""
    if a is None or b is None:
        return None
    return a - b



def profile_generation(generate: Callable):
    @wraps(generate)
    def wrapper(
        self: object, prompt: str, generation_config: GenConfig, *args, **kwargs
    ) -> GenerationResponse:

        gen_start_time: float = time.time()
        # Execute the wrapped function
        response: GenerationResponse = generate(
            self, prompt, generation_config, *args, **kwargs
        )

        gen_end_time: float = time.time()

        perf_metrics: PerfMetrics = PerfMetrics.from_timestamps(
            start_time=gen_start_time,
            grammar_compilation_end_time=response.metadata._grammar_compilation_end_time,
            first_token_arrival_time=response.metadata._first_tok_arr_time,
            end_time=gen_end_time,
            num_output_tokens=response.token_usage.output_tokens,  # should we use no_ff_output_tokens here?
        )

        response.perf_metrics = perf_metrics

        return response

    return wrapper