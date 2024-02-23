- [Introduction](#introduction)
- [Model download](#download)
- [Run the model](#inference)
- [Fine-tuning the model](#finetuning)
- [Limitations](#limitations)

# PhoGPT: Generative Pre-training for Vietnamese <a name="introduction"></a>


We open-source a state-of-the-art 4B-parameter generative model series for Vietnamese, which includes the base pre-trained monolingual model PhoGPT-4B and its chat variant, PhoGPT-4B-Chat. The base model, PhoGPT-4B, with exactly 3.7B parameters, is pre-trained from scratch on a Vietnamese corpus of 102B tokens, with an 8192 context length, employing a vocabulary of 20K token types. The chat variant, PhoGPT-4B-Chat, is the modeling output obtained by fine-tuning PhoGPT-4B on a dataset of 70K instructional prompts and their responses, along with an additional 290K conversations. We demonstrate its strong performance compared to previous closed-source and open-source 7B-parameter models. 


More details about the general architecture and experimental results of PhoGPT can be found in our [technical report](https://arxiv.org/abs/2311.02945). All output responses of PhoGPT and baselines are available [HERE](https://docs.google.com/spreadsheets/d/1R228Fnrwo4d2PSEJlgHdr9Q49zWWvz3k7pflw0pNTpo/edit?usp=sharing) for readers' self-evaluation. **Please CITE** our technical report when PhoGPT is used to help produce published results or is incorporated into other software:

```
@article{PhoGPT,
title     = {{PhoGPT: Generative Pre-training for Vietnamese}},
author    = {Dat Quoc Nguyen and Linh The Nguyen and Chi Tran and Dung Ngoc Nguyen and Dinh Phung and Hung Bui},
journal   = {arXiv preprint},
volume    = {arXiv:2311.02945},
year      = {2023}
}
```


## Model download <a name="download"></a>

Model | Type | Model Size | Context length | Vocab size | Training data size | Note
---|--|---|---|---|---|---
[`vinai/PhoGPT-4B-v0.1`](https://huggingface.co/vinai/PhoGPT-4B-v0.1) | Base | 3.7B | 8192 | 20K | 482GB of texts | Loading "PhoGPT-4B-v0.1" or "PhoGPT-4B-Chat-v0.1" in float16 takes 7GB of GPU memory
[`vinai/PhoGPT-4B-Chat-v0.1`](https://huggingface.co/vinai/PhoGPT-4B-Chat-v0.1) |Instruction following & Chat|3.7B| 8192| 20K |70K instructional prompt and response pairs & 290K conversations| `PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"`  

## Run the model <a name="inference"></a>

### with vLLM, Text Generation Inference & llama.cpp

PhoGPT can run with inference engines, such as [vLLM](https://github.com/vllm-project/vllm), [Text Generation Inference](https://github.com/huggingface/text-generation-inference) and [llama.cpp](https://github.com/ggerganov/llama.cpp).

### with pure `transformers`

#### Instruction following

```python
# coding: utf8
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_path = "vinai/PhoGPT-4B-Chat-v0.1"  

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  
config.init_device = "cuda"

model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
# If your GPU does not support bfloat16:
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True)
model.eval()  

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  

PROMPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"  

# Some instruction examples
# instruction = "Viết bài văn nghị luận xã hội về {topic}"
# instruction = "Viết bản mô tả công việc cho vị trí {job_title}"
# instruction = "Sửa lỗi chính tả:\n{sentence_or_paragraph}"
# instruction = "Dựa vào văn bản sau đây:\n{text}\nHãy trả lời câu hỏi: {question}"
# instruction = "Tóm tắt văn bản:\n{text}"

instruction = "Viết bài văn nghị luận xã hội về an toàn giao thông"
# instruction = "Sửa lỗi chính tả:\nTriệt phá băng nhóm kướp ô tô, sử dụng \"vũ khí nóng\""

input_prompt = PROMPT_TEMPLATE.format_map({"instruction": instruction})  

input_ids = tokenizer(input_prompt, return_tensors="pt")  

outputs = model.generate(  
    inputs=input_ids["input_ids"].to("cuda"),  
    attention_mask=input_ids["attention_mask"].to("cuda"),  
    do_sample=True,  
    temperature=1.0,  
    top_k=50,  
    top_p=0.9,  
    max_new_tokens=1024,  
    eos_token_id=tokenizer.eos_token_id,  
    pad_token_id=tokenizer.pad_token_id  
)  

response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
response = response.split("### Trả lời:")[1]
```

#### Chat

```python
messages = [
    {"role": "user", "content": "Kể tên một môn thể thao mạo hiểm"},
    {"role": "assistant", "content": "Nhảy Bungee."},
    {"role": "user", "content": "Bạn đã bao giờ đi nhảy bungee chưa"}
]

# Using apply_chat_template
tokenizer = AutoTokenizer.from_pretrained("vinai/PhoGPT-4B-Chat-v0.1", trust_remote_code=True)
input_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

## Fine-tuning the model <a name="finetuning"></a>

See [llm-foundry docs](https://github.com/mosaicml/llm-foundry/blob/main/scripts/train/README.md#llmfinetuning) for details. To fully fine-tune PhoGPT, users can find an example of model finetuning YAML configuration at [`fine-tuning-phogpt.yaml`](https://github.com/VinAIResearch/PhoGPT/blob/main/fine-tuning-phogpt.yaml). Users can also find the `sample_instruction_following_dataset` folder as an example of an instruction-following dataset.

- To install `llm-foundry`, see Section "Installation" in [https://github.com/mosaicml/llm-foundry](https://github.com/mosaicml/llm-foundry).
- Run: `cd llm-foundry/scripts/train/` and then `composer --world_size <number_of_GPUs> train.py <path_to_yaml_configuration_file>` (e.g. `composer --world_size 1 train.py fine-tuning-phogpt.yaml`). 

## Limitations <a name="limitations"></a>

PhoGPT has certain limitations. For example, it is not good at tasks involving reasoning, coding or mathematics. PhoGPT may generate harmful, hate speech, biased responses, or answer unsafe questions. Users should be cautious when interacting with PhoGPT that can produce factually incorrect output.

## License

```
Copyright (c) 2023 VinAI Research

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
