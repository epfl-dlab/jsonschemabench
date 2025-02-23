import os

from omegaconf import OmegaConf

from jsonschemabench.engine import BaseEngine
from jsonschemabench.utils import CONFIG_DIR


# llama-cpp backend

openai_model_cfg = OmegaConf.load(
    os.path.join(CONFIG_DIR, "model", "gemini", "gemini-1.5-pro.yaml")
)

model = BaseEngine.create(openai_model_cfg)


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
    os.path.join(CONFIG_DIR, "generation", "gemini", "free_gen_code.yaml")
)

generation_cfg = model.config_cls.create(raw_generation_cfg)


response = model.generate(prompt, generation_config=generation_cfg)

print(f"Free form generation: {response.output}")

# json mode generation

raw_generation_cfg = OmegaConf.load(
    os.path.join(CONFIG_DIR, "generation", "gemini", "json_mode.yaml")
)

generation_cfg = model.config_cls.create(raw_generation_cfg)

response = model.generate(prompt, generation_config=generation_cfg)

print(f"JSON mode generation: {response.output}")


# json schema mode generation

raw_generation_cfg = OmegaConf.load(
    os.path.join(CONFIG_DIR, "generation", "gemini", "json_schema.yaml")
)

generation_cfg = model.config_cls.create(raw_generation_cfg)

response = model.generate(
    prompt,
    generation_config=generation_cfg,
    schema="examples/JSON_schema/city_info.schema.json",
)

print(f"JSON schema mode generation: {response.output}")
