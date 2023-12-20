import json
  
import datasets
import os
import logging
import lm_dataformat
from tqdm import tqdm

try:
    import lm_dataformat
except ImportError:
    print(
        "Can't import lm_dataformat, please run pip install lm_dataformat and try again"
    )
    exit()

_URL = "path/to/sample_instruction_following_dataset/sample_prompt_response_pairs.jsonl"

class Vinewstext(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="plain_text",
            description="Plain text",
            version=datasets.Version("1.0.0"),
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="",
            features=datasets.Features({"prompt": datasets.Value("string"), "response": datasets.Value("string")}),
            homepage="",
            citation="",
        )

    def _split_generators(self, dl_manager):
        downloaded_file = dl_manager.download(_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"file_path": downloaded_file}),
        ]

    def _generate_examples(self, file_path):
        logging.warning(f"Generating examples from {file_path}")
        _id = 0
        with open(file_path, encoding='utf-8') as json_file:
            for line in json_file:
                doc = json.loads(line)
                yield _id, {"prompt": doc["prompt"], "response": doc["response"]}
                _id += 1
