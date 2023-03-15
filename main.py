import pydantic

from fastapi import FastAPI
from inference import load
import os

from llama.generation import LLaMA

app = FastAPI()


class Input(pydantic.BaseModel):
    prompts: list[str] = []
    temperature: float = 0.8
    top_p: float = 0.95


class Output(pydantic.BaseModel):
    results: list[str]


_generator: LLaMA = None


@app.on_event("startup")
def preload():
    global _generator
    local_rank = 0
    world_size = 1
    max_seq_len = 1024
    max_batch_size = 1

    _generator = load(
        os.environ["ckpt_dir"],
        os.environ["tokenizer_path"],
        local_rank,
        world_size,
        max_seq_len,
        max_batch_size,
    )


@app.post("/prompt")
def prompt(_input: Input) -> Output:
    results = _generator.generate(
        _input.prompts,
        max_gen_len=256,
        temperature=_input.temperature,
        top_p=_input.top_p,
    )
    return Output(results=results)
