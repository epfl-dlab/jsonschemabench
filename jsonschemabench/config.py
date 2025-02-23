from typing import Union
import copy
from omegaconf import OmegaConf, DictConfig
from transformers.generation import GenerationConfig
import json, os
import logging
logger = logging.getLogger(__name__)


class GenConfig(object):
    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def create(cfg) -> "GenConfig":

        if "cls" not in cfg:
            cfg = cfg.generation

        class_mapping = {
            "Guidance": GuidanceGenConfig,
            "HF": HFGenConfig,
            "OpenAI": OpenaiGenConfig,
            "Gemini": GeminiGenConfig,
            "Outlines": OutlinesGenConfig,
            "LlamaCpp": LlamaCppGenConfig,
            "XGrammar": XGrammarGenConfig,
        }

        cls = class_mapping.get(cfg.cls)
        return cls.init(cfg)

    @staticmethod
    def init(cfg):
        raise NotImplementedError

    @staticmethod
    def load_json_schema(json_schema: Union[str, dict]) -> dict:
        if json_schema is None:
            return None
        if isinstance(json_schema, DictConfig):
            json_schema = OmegaConf.to_container(json_schema)
        if isinstance(json_schema, dict):
            json_schema_obj = json_schema
            return json_schema_obj
        elif isinstance(json_schema, str):
            # striping is necessary to remove the whitespace and newline at the end of the string
            json_schema = json_schema.strip()
            if json_schema.startswith("{") and json_schema.endswith("}"):
                # see if this is a json schema string
                json_schema_obj = json.loads(json_schema)
            elif os.path.exists(json_schema):
                with open(json_schema, "r") as f:
                    json_schema_obj = json.load(f)
            else:
                raise ValueError(f"json_schema file {json_schema} not found")
            return json_schema_obj
        else:
            raise ValueError("json_schema must be a string or a dict")

    def copy(self) -> "GenConfig":
        # Perform a deep copy of the instance
        return copy.deepcopy(self)

    def update_json_schema(self, json_schema: Union[str, dict]) -> "GenConfig":
        raise NotImplementedError


class HFGenConfig(GenConfig):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gen_config = GenerationConfig.from_pretrained(
            cfg.pretrained_model_name_or_path
        )
        extra_gen_cfg_dict = dict(self.cfg.kwargs)
        self.gen_config.update(**extra_gen_cfg_dict)

    @staticmethod
    def init(cfg):
        return HFGenConfig(cfg)


class XGrammarGenConfig(GenConfig):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gen_config = OmegaConf.create(cfg)
        if self.gen_config.mode == "json_schema":
            self.gen_config.json_schema = self.load_json_schema(
                self.gen_config.json_schema
            )

    @staticmethod
    def init(cfg):
        return XGrammarGenConfig(cfg)

    def update_json_schema(self, json_schema: Union[str, dict]) -> GenConfig:
        # check if the config is in json schema mode
        if self.gen_config.mode != "json_schema":
            logger.warn(
                f"Generation is not in json schema mode, the json schema will be ignored"
            )

        copy_config = self.copy()
        # TODO should json_schema be a string or a dict?
        json_schema_obj = self.load_json_schema(json_schema)
        copy_config.gen_config.json_schema = json_schema_obj  # ["parameters"]
        return copy_config


class GuidanceGenConfig(GenConfig):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gen_config = OmegaConf.create(cfg)
        if self.gen_config.mode == "json_schema":
            self.gen_config.operator.kwargs.json_schema = self.load_json_schema(
                self.gen_config.json_schema
            )

    @staticmethod
    def init(cfg):
        return GuidanceGenConfig(cfg)

    def update_json_schema(self, json_schema: Union[str, dict]) -> GenConfig:
        # check if the config is in json schema mode
        if self.gen_config.mode != "json_schema":
            logger.warn(
                f"Generation is not in json schema mode, the json schema will be ignored"
            )

        copy_config = self.copy()
        # TODO should json_schema be a string or a dict?
        json_schema_obj = self.load_json_schema(json_schema)
        copy_config.gen_config.operator.kwargs.json_schema = (
            json_schema_obj  # ["parameters"]
        )
        return copy_config


class OpenaiGenConfig(GenConfig):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gen_config = OmegaConf.create(cfg)
        if self.gen_config.mode == "json_schema":
            self.gen_config.kwargs.response_format.json_schema.schema = (
                self.load_json_schema(self.gen_config.json_schema)
            )

    @staticmethod
    def init(cfg):
        return OpenaiGenConfig(cfg)

    def update_json_schema(self, json_schema: Union[str, dict]) -> "OpenaiGenConfig":
        # check if the config is in json schema mode
        if self.gen_config.mode != "json_schema":
            logger.warn(
                f"Generation is not in json schema mode, the json schema will be ignored"
            )
        # return a deep copy of the object
        copy_config = self.copy()
        # TODO should json_schema be a string or a dict?
        json_schema_obj = self.load_json_schema(json_schema)
        copy_config.gen_config.kwargs.response_format.json_schema.schema = (
            json_schema_obj
        )
        return copy_config


class GeminiGenConfig(GenConfig):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gen_config = OmegaConf.create(cfg)
        if self.gen_config.mode == "json_schema":
            self.gen_config.kwargs.response_schema = self.load_json_schema(
                self.gen_config.json_schema
            )

    @staticmethod
    def init(cfg):
        return GeminiGenConfig(cfg)

    def update_json_schema(self, json_schema: Union[str, dict]) -> GenConfig:
        # check if the config is in json schema mode
        if self.gen_config.mode != "json_schema":
            logger.warn(
                f"Generation is not in json schema mode, the json schema will be ignored"
            )
        # return a deep copy of the object
        copy_config = self.copy()
        # TODO should json_schema be a string or a dict?
        json_schema_obj = self.load_json_schema(json_schema)
        copy_config.gen_config.kwargs.response_schema = json_schema_obj
        return copy_config


class OutlinesGenConfig(GenConfig):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gen_config = OmegaConf.create(cfg)
        if self.gen_config.mode == "json_schema":
            self.gen_config.json_schema = self.load_json_schema(
                self.gen_config.json_schema
            )

    @staticmethod
    def init(cfg):
        return OutlinesGenConfig(cfg)

    def update_json_schema(self, json_schema: Union[str, dict]) -> GenConfig:
        # check if the config is in json schema mode
        if self.gen_config.mode != "json_schema":
            logger.warn(
                f"Generation is not in json schema mode, the json schema will be ignored"
            )
        # return a deep copy of the object
        copy_config = self.copy()
        # TODO should json_schema be a string or a dict?
        json_schema_obj = self.load_json_schema(json_schema)
        copy_config.gen_config.json_schema = json_schema_obj
        return copy_config


class LlamaCppGenConfig(GenConfig):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.gen_config = OmegaConf.create(cfg)
        if self.gen_config.mode == "json_schema":
            self.gen_config.kwargs.response_format.schema = self.load_json_schema(
                self.gen_config.json_schema
            )

    @staticmethod
    def init(cfg):
        return LlamaCppGenConfig(cfg)

    def update_json_schema(self, json_schema: Union[str, dict]) -> GenConfig:
        # check if the config is in json schema mode
        if self.gen_config.mode != "json_schema":
            logger.warn(
                f"Generation is not in json schema mode, the json schema will be ignored"
            )
        # return a deep copy of the object
        copy_config = self.copy()
        # TODO should json_schema be a string or a dict?
        json_schema_obj = self.load_json_schema(json_schema)
        copy_config.gen_config.kwargs.response_format.schema = json_schema_obj
        return copy_config
