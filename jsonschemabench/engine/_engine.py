from abc import ABC, abstractmethod
import inspect
from typing import Dict, List, Optional, Union

from ..config import GenConfig
from ..utils import GenerationResponse


FRAMEWORK_NAMES = {
    "llama_cpp": "llama.cpp",
    "openai": "OpenAI",
    "guidance": "Guidance",
    "hf": "Hugging Face",
    "outlines": "Outlines",
    "gemini": "Google-Gemini",
    "xgrammar": "XGrammar",
}


class BaseEngine(ABC):

    config_cls = GenConfig

    def __init__(self, cfg):
        self.cfg = cfg
        self.total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "num_samples": 0,
        }
        self.grammar_engine_version = None

    # Core Abstract Methods
    # --------------------

    @abstractmethod
    def generate(
        self,
        prompt: str,
        generation_config: GenConfig,
        schema: Optional[str] = None,
    ) -> GenerationResponse:
        """Generate text based on the input prompt.

        Args:
            prompt: Input text to generate from
            generation_config: Configuration for text generation

        Returns:
            GenerationResponse containing the generated text and metadata
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def max_ctx_len(self) -> int:
        """Maximum context length supported by the model.

        Returns:
            Integer representing maximum number of tokens
        """
        raise NotImplementedError

    # Factory Methods
    # --------------

    @staticmethod
    def create(cfg) -> "BaseEngine":
        if "model" in cfg:
            cfg = cfg.model

        # Dynamically get all model classes
        from jsonschemabench import models

        _classes: List[tuple[str, type]] = inspect.getmembers(
            models,
            lambda x: inspect.isclass(x)
            and issubclass(x, BaseEngine)
            and x is not BaseEngine,
        )

        # Convert to a dictionary for quick lookup
        classes: Dict[str, type] = {cls.__name__: cls for _, cls in _classes}

        # the second argument is for backward compatibility prior to v0.2.0
        model_cls = classes.get(cfg.cls) or classes.get(f"{cfg.cls}Model")
        assert model_cls, f"Model class {cfg.cls} not found in {list(classes.keys())}"

        return model_cls.init(cfg)

    @staticmethod
    def init(cfg):
        """Initialize a model instance with given config.

        To be implemented by subclasses.
        """
        raise NotImplementedError

    # Input Validation
    # ---------------

    def validate_input(self, prompt: str) -> bool:
        """Validate if input prompt is within model's context window.

        Args:
            prompt: Input text to validate

        Returns:
            Boolean indicating if input is valid
        """
        max_ctx_window = self.max_ctx_len
        # Margin necessary to leave space for generation
        margin = 0.2
        input_tokens = self.count_tokens(prompt)
        return input_tokens <= max_ctx_window * (1 - margin)

    # Tokenization Methods
    # -------------------

    def encode(self, text: str) -> Union[List[int], None]:
        """Convert text to token IDs.

        Args:
            text: Input text to encode

        Returns:
            List of token IDs or None if encoding not supported
        """
        return None

    def decode(self, ids: List[int]) -> Union[str, None]:
        """Convert token IDs back to text.

        Args:
            ids: List of token IDs to decode

        Returns:
            Decoded text or None if decoding not supported
        """
        return None

    def convert_token_to_id(self, token: str) -> int:
        """Convert a single token to its ID.

        Args:
            token: Token string to convert

        Returns:
            Integer ID of the token
        """
        res = self.encode(token)
        return (
            res[0] if len(res) == 1 else None
        )  # return None if token is not found, res is empty list

    def convert_id_to_token(self, id: int) -> str:
        """Convert a single token ID to its string representation.

        Args:
            id: Token ID to convert

        Returns:
            String representation of the token
        """
        return self.decode([id])[0]

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text.

        Args:
            text: Input text to count tokens for

        Returns:
            Number of tokens in text
        """
        return len(self.encode(text))