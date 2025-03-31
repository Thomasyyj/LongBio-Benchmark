# LongBio Benchmark: A controlled benchmark for Long-Context Language Models

LongBio benchmark generate a coherent context and comprehensive questions to evaluate long-conotext language models with controlled configuations and tasks with increasing difficulty levels. For context, we allowed several controllable features (context length, distractor density, the number of answer information and their position) while ensuring the coherance of texts. For questions, we design 15 tasks covering the the capbilities of 1\) understanding, 2\) reasoning and 3\) trustworthy generations of LLMs and make the sub-task hierarchical to positional the level of each LLMs. Here are the main results. 

|                                  |   Understanding |   Reasoning |   Trustworthy |    avg |   rank |
|:---------------------------------|----------------:|------------:|--------------:|-------:|-------:|
| Qwen2.5-14B-Instruct-1M          |            90.8 |        47.6 |          63.8 | 67.400 |      1 |
| Qwen2.5-7B-Instruct-1M           |            86.2 |        34.9 |          71.8 | 64.300 |      2 |
| internlm3-8b-instruct            |            68.3 |        28.6 |          65.4 | 54.100 |      3 |
| glm-4-9b-chat-1m                 |            54.9 |        33.2 |          55.2 | 47.767 |      4 |
| Llama-3-8B-ProLong-512k-Instruct |            45.4 |        30.0 |          62.2 | 45.867 |      5 |
| Llama-3.1-8B-Instruct            |            50.3 |        17.9 |          59.0 | 42.400 |      6 |
| Llama-3.1-70B-Instruct           |            32.2 |        18.5 |          29.1 | 26.600 |      7 |
| Qwen2.5-72B-Instruct             |            33.4 |        24.2 |           7.6 | 21.733 |      8 |
| Phi-3.5-mini-instruct            |            25.3 |        10.2 |          21.9 | 19.133 |      9 |
| Llama-3.2-3B-Instruct            |            13.1 |        13.9 |          16.2 | 14.400 |     10 |
| Qwen2.5-7B-Instruct              |            15.5 |        13.0 |          14.6 | 14.367 |     11 |
| Phi-3-medium-128k-instruct       |            16.9 |         8.4 |           1.8 |  9.033 |     12 |
| Llama-3.2-1B-Instruct            |             3.0 |         9.0 |           4.4 |  5.467 |     13 |
| Mistral-Nemo-Instruct-2407       |             2.5 |         7.0 |           0.0 |  3.167 |     14 |

We are continuously updating the results.


## Requirements

Install the package:
```
pip install -r requirements.txt
```

Before running the script, please specify the model path and the home path using
```
# the home directory of all models, one should have models like ${home}/Qwen2.5-7B
export MODEL_PATH="path/to/folder/of/all/models"
# the home directory of storing this repo
export HOME="path/to/this/repo"
```

## Create LongBio Dataset
To create the dataset, run
```
bash scripts/submit_scripts/prepare_data.sh
```
If you want, you can modify the model family of the tokenizer for creating the data in the shell file. The default tokenizer is Qwen2.5-7B.

## Run experiments
To run the experiments (e.g calculation task) on a specific model, 
```
export MODEL_NAME="folder/name/of/model" # the folder name of the model you want to use under MODEL_PATH 
export TOKENIZER_NAME="Qwen2.5-7B" # should be the same as the name you use when constructing the data 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/submit_scripts/task_qwen/calculation.sh # You must specify the available gpu devices
```

## Results
The results and the logs will be stored at "inference_results/" automatically:
```
DATA_NAME/
├── config.json              # The experiment results (Accuracy) and the config of experiment
├── data.json                # The data used for testing
├── inference results.json   # The output results from llm
└── terminal.log             # The logs of the scripts
```
