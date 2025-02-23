import os

from omegaconf import OmegaConf

from jsonschemabench.engine import BaseModel
from jsonschemabench.utils import CONFIG_DIR


# llama-cpp backend

openai_model_cfg = OmegaConf.load(
    os.path.join(CONFIG_DIR, "model", "openai", "gpt-4o-2024-08-06.yaml")
)

model = BaseModel.create(openai_model_cfg)


#################
#
# 3 different generation modes
#
#################

text = """
Paris (French pronunciation: [paʁi] ⓘ) is the capital and largest city of France. With an estimated population of 2,102,650 residents in January 2023[2] in an area of more than 105 km2 (41 sq mi),[5] Paris is the fourth-most populous city in the European Union, the ninth-most populous city in Europe and the 30th most densely populated city in the world in 2022
"""

prompt = f"Extract information from the following text: {text} as JSON"

# free form generation

raw_generation_cfg = OmegaConf.load(
    os.path.join(CONFIG_DIR, "generation", "openai", "free_gen_code.yaml")
)

generation_cfg = model.config_cls.create(raw_generation_cfg)


response = model.generate(prompt, generation_config=generation_cfg)

print(f"Free form generation: {response.output}")


# json mode generation
# TODO broken

# raw_generation_cfg = OmegaConf.load(
#     os.path.join(CONFIG_DIR, "generation", "openai", "json_mode.yaml")
# )

# raw_generation_cfg.generation.kwargs.max_tokens = 10

# generation_cfg = model.config_cls.create(raw_generation_cfg)

# response = model.generate(prompt, generation_config=generation_cfg)

# print(response.output)


# json schema mode generation

raw_generation_cfg = OmegaConf.load(
    os.path.join(CONFIG_DIR, "generation", "openai", "json_schema.yaml")
)

raw_generation_cfg.generation.kwargs.max_tokens = 10

generation_cfg = model.config_cls.create(raw_generation_cfg)

response = model.generate(
    prompt,
    generation_config=generation_cfg,
    schema="examples/JSON_schema/city_info.schema.json",
)

print(f"JSON schema mode generation: {response.output}")
