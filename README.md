# LongBio Benchmark: A controlled benchmark for Long-Context Language Models

LongBio benchmark generate a coherent context and comprehensive questions to evaluate long-conotext language models with controlled configuations and tasks with increasing difficulty levels. For context, we allowed several controllable features (context length, distractor density, the number of answer information and their position) while ensuring the coherance of texts. For questions, we design 15 tasks covering the the capbilities of 1\) understanding, 2\) reasoning and 3\) trustworthy generations of LLMs and make the sub-task hierarchical to positional the level of each LLMs. Here are the main results. 

|           128k                   |   Understanding |   Reasoning |   Trustworthy |    avg |   rank |
|:---------------------------------|----------------:|------------:|--------------:|-------:|-------:|
| gpt-4o-2024-11-20                |            98.0 |        66.5 |          54.6 | 73.033 |      1 |
| Qwen2.5-14B-Instruct-1M          |            91.7 |        58.0 |          63.8 | 71.167 |      2 |
| gpt-4o-mini-2024-07-18           |            71.2 |        39.6 |          88.8 | 66.533 |      3 |
| Qwen2.5-7B-Instruct-1M           |            87.1 |        34.9 |          71.8 | 64.600 |      4 |
| internlm3-8b-instruct            |            68.6 |        28.6 |          65.4 | 54.200 |      5 |
| glm-4-9b-chat-1m                 |            55.6 |        34.3 |          55.2 | 48.367 |      6 |
| Llama-3-8B-ProLong-512k-Instruct |            45.8 |        31.7 |          62.2 | 46.567 |      7 |
| Llama-3.1-8B-Instruct            |            50.5 |        16.8 |          59.0 | 42.100 |      8 |
| Llama-3.1-70B-Instruct           |            32.2 |        17.6 |          29.1 | 26.300 |      9 |
| Phi-3.5-mini-instruct            |            25.3 |        28.4 |          21.9 | 25.200 |     10 |
| Qwen2.5-72B-Instruct             |            33.6 |        32.0 |           7.6 | 24.400 |     11 |
| Qwen2.5-7B-Instruct              |            15.4 |        23.0 |          14.6 | 17.667 |     12 |
| Llama-3.2-3B-Instruct            |            13.2 |        17.2 |          16.2 | 15.533 |     13 |
| Llama-3.3-70B-Instruct           |             9.8 |        12.9 |           8.3 | 10.333 |     14 |
| Llama-3.2-1B-Instruct            |             3.1 |        19.8 |           4.4 |  9.100 |     15 |
| Phi-3-medium-128k-instruct       |            16.9 |         6.8 |           1.8 |  8.500 |     16 |
| Mistral-Nemo-Instruct-2407       |             2.4 |         7.2 |           0.0 |  3.200 |     17 |

|            64k                   |   Understanding |   Reasoning |   Trustworthy |    avg |   rank |
|:---------------------------------|----------------:|------------:|--------------:|-------:|-------:|
| Qwen2.5-14B-Instruct-1M          |            96.3 |        60.3 |          73.6 | 76.733 |      1 |
| Llama-3.1-70B-Instruct           |            90.9 |        49.7 |          89.0 | 76.533 |      2 |
| Llama-3.3-70B-Instruct           |            84.3 |        61.7 |          81.8 | 75.933 |      3 |
| Llama-3.1-8B-Instruct            |            85.6 |        40.1 |          90.0 | 71.900 |      4 |
| internlm3-8b-instruct            |            84.0 |        35.7 |          77.5 | 65.733 |      5 |
| Qwen2.5-7B-Instruct-1M           |            94.0 |        38.6 |          60.2 | 64.267 |      6 |
| glm-4-9b-chat-1m                 |            75.6 |        40.3 |          72.8 | 62.900 |      7 |
| Llama-3-8B-ProLong-512k-Instruct |            57.4 |        34.6 |          82.2 | 58.067 |      8 |
| Phi-3.5-mini-instruct            |            62.9 |        46.0 |          46.2 | 51.700 |      9 |
| Phi-3-medium-128k-instruct       |            61.1 |        28.2 |          55.0 | 48.100 |     10 |
| Qwen2.5-72B-Instruct             |            50.0 |        40.1 |          16.6 | 35.567 |     11 |
| Qwen2.5-7B-Instruct              |            36.1 |        32.2 |          33.5 | 33.933 |     12 |
| Llama-3.2-3B-Instruct            |            23.0 |        26.0 |          27.0 | 25.333 |     13 |
| Llama-3.2-1B-Instruct            |             9.4 |        15.8 |          10.4 | 11.867 |     14 |
| Mistral-Nemo-Instruct-2407       |             9.5 |         8.7 |           9.6 |  9.267 |     15 |

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

## Task Description

The tasks to be evaluated on are divided in to three categories: Understanding, Reasoning and Trustworthy

### Understanding

| Task                      | Description                                                  | Default Config                           | Metric                  | Examples                                                     |
| ------------------------- | ------------------------------------------------------------ | ---------------------------------------- | ----------------------- | ------------------------------------------------------------ |
| standard                  | The standard version of the retrieval task. The llm will be asked to retrieve a specific attribute of one person. | n_retrieval=<br />1;<br />Zero-shot      | Accuracy                | Attribute: Below is the bio of {P1}. ... The hobby of {P1} is dandyism<br />Question: What's the hobby of {P1}? |
| multi_standard (standard) | The extended standard version. The llm will be asked to retrieve multiple attributes of different people. | n_retrieval=<br />2,5,10;<br />Zero-shot | All-or-Nothing Accuracy | Attribute: Below is the bio of {P1}. ... The hobby of {P1} is dandyism. ... Below is the bio of {P2}. ... The hobby of {P2} is mycology. ...<br />Question: Question1: What's the hobby of {P1}? Question2: What's the hobby of {P2}? |
| paraphrase                | The upgrade of the standard version, where the expressions of attributes in the context are replaced by its paraphrases | n_retrieval=<br />1;<br />Zero-shot      | Accuracy                | Attribute: Below is the bio of {P1}. ... {P1} enhanced his/her vocational skills by working in the city of Dhaka.<br />Question: Which city did {P1} work in? |
| pronoun                   | The upgrade of the paraphrase version, where each bio is expressed from a first-person perspective | n_retrieval=<br />1;<br />Zero-shot      | Accuracy                | Attribute: Below is the self-introduction of {P1}. I was born on the day of 1993-06-26.<br />Question: What is the birthday of {P1}? |



### Reasoning

| Task        | Description                                                  | Default Config                        | Metric   | Examples                                                     |
| ----------- | ------------------------------------------------------------ | ------------------------------------- | -------- | ------------------------------------------------------------ |
| calculation | The upgrade of the multi_standard version, where the model is asked to give the age difference between two given people. | n_retrieval=<br />2;<br />Two-shot    | Accuracy | Attribute: Below is the bio of {P1}. ...The age of {P1} is 61. ... Below is the bio of {P2}. The age of {P2} is 43...<br />Question: What’s the age difference between {P1} and {P2}? |
| rank        | The upgrade of the multi_standard version, where the model is asked to rank the given people based on their ages. | n_retrieval=<br />2, 5;<br />Two-shot | Accuracy | Attribute: Below is the bio of {P1}. ...The age of {P1} is 61. ... Below is the bio of {P2}. The age of {P2} is 43...<br />Question: Rank the following people in order of age from the youngest to the oldest: {P1} and {P2} |
| multihop    | The upgrade of the multi_standard version, where the model is asked to retrieve the attribute of a person by tracking the attributes between different people. | n_retrieval=<br />2, 5;<br />Two-shot | Accuracy | Attribute: Below is the bio of {P1}. ... {P1} was born in Santa Paula. ...Below is the bio of {P2}. ... The birthplace of {P2} is the same as Owen Xanthus Jimenez.<br />Question: What is the birthplace of {P2}? |
| twodiff     | The upgrade of the calculation and rank version, where the model is asked to give the name of two people whose ages difference are the given specific number. | n_retrieval=<br />2;<br />Two-shot    | Accuracy | Attribute: Below is the bio of {P1}. ...The age of {P1} is 61. ... Below is the bio of {P2}. The age of {P2} is 43...<br />Question: From the given profiles of these individuals, identify two individuals such that the age difference between them is 18. Only one pair of answers is needed. |

### Trustworthy

| Task     | Description                                                  | Default Config                       | Metric                                                | Examples                                                     |
| -------- | ------------------------------------------------------------ | ------------------------------------ | ----------------------------------------------------- | ------------------------------------------------------------ |
| citation | The upgrade of the multi_standard version, where the model is asked to give the answer as well as the citation of this answer. | n_retrieval=<br />1,2;<br />Two-shot | Answer Accuracy;<br />Citation Accuracy               | Attribute: Bio [1]: ... {P1} was born in Santa Paula.<br />Question: Which university did Isabel Winston Nolan graduate from? |
| idk      | The upgrade of the standard version, where the correct answer is deleted. The model should output "The answer is not explicitly stated" when the answer is deleted and answer the correct answer when the answer is given | n_retrieval=<br />1;<br />Zero-shot  | Answer Accuracy;<br />Refuse Accracy<br />All-Accracy | Attribute: Below is the bio of {P1}. ... (delete the expression of hobby)<br />Question: What's the hobby of {P1}? |

## Citation
```
@misc{longbio2025,
  title = {LongBio-Benchmark: A Controlled Benchmark for Long-Context Language Models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
}
```