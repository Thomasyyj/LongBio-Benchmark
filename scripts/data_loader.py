import json
import os
import re
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from loguru import logger

from scripts.data_loader_prompts import PROMPT_TEMPLATE
from scripts.utils import fill_template

class LongContextDataSetFactory:
    def __new__(cls, args, data_configs, **kwargs):
        tasks_map = {
            "SingleQA": LongContextDataSet,
            "IDK": LongContextDataSetForIDK,
            "MultiQA": LongContextDataSetForMultiQA,
            "Rank": LongContextDataSetForRank,
            "Calculation": LongContextDataSetForCal,
            "TwoDiff": LongContextDataSetForTwoDiff,
            "Multihop": LongContextDataSetForMultihop,
            "FewShotSingleQA": FewShotDataSet,
            "FewShotMultiQA": FewShotDataSetForMultiQA,
            "FewShotRank": FewShotDataSetForRank,
            "FewShotCalculation": FewShotDataSetForCal,
            "FewShotTwoDiff": FewShotDataSetForTwoDiff,
            "FewShotMultihop": FewShotDataSetForMultihop,
        }

        if data_configs["skill"] == "IDK":
            skill = "IDK"
        elif data_configs["num_retrieval"] == 1:
            skill = "SingleQA"
        else:
            if data_configs["skill"] == "rank":
                skill = "Rank"
            elif data_configs["skill"] == "calculation":
                skill = "Calculation"
            elif data_configs["skill"] == "twodiff":
                skill = "TwoDiff"
            elif data_configs["skill"] == "multihop":
                skill = "Multihop"
            else:
                skill = "MultiQA"
        
        if args.apply_fewshot:
            skill = "FewShot" + skill

        return tasks_map[skill](**kwargs)


class LongContextDataSet(Dataset):
    def __init__(self, model_path, data_path, skill="SingleQA"):
        """
        JSON
        [
            {
                "context": [str],
                "refrence": [str],  # no need
                "tok_length": int,
                "prefix": 
                "questions":{
                    "QA":[
                        "Question":str,
                        "Answer":str,
                        "Reference": int,
                        "attribution": str,
                        "position":List((int, int)()
                    ]
                }
            }
        ]
        """
        
        self.model_path = model_path
        self.data_path = data_path
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.full_data = json.load(f)
        if 'gpt' not in model_path:
            with open(os.path.join(model_path, "config.json"), 'r') as f:
                self.model_config = json.load(f)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.tokenizer = 'gpt'
        self.skill = skill
        self.template = self.prepare_prompt_template()
        self.data = None
    
    @staticmethod
    def trim_string_until_target(s, t):
        while not s.endswith(t):
            s = s[:-1]
        return s
        
    def prepare_prompt_template(self):

        messages = PROMPT_TEMPLATE[self.skill]
        if self.tokenizer == 'gpt':
            return messages
        if "instruct" in self.model_path.lower() or "chat" in self.model_path.lower():
            prompt_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
            logger.info(f"{self.skill} Prompt template applied: {messages}")
        else:
            logger.warning("No chat template find. Try to use template for base model")
            if self.skill + "ForBase" in PROMPT_TEMPLATE.keys():
                messages = PROMPT_TEMPLATE[self.skill + "ForBase"]
                logger.info(f"{self.skill}ForBase Prompt template applied: {messages}")
                return messages
            else:
                assert False, "Current task did not support base model. Please turn to fewshot case"
        
        prompt_template = self.trim_string_until_target(prompt_template, messages[-1]["content"].strip())
        return prompt_template

    def split_question(self, questions):
        '''
        
        'Question1: The major of Evelyn Iver Smith was
        Question2: The major of Landon Jethro Lambert was
        Question3: The major of Evelyn Zelda Lucas was'

        -> "The major of Evelyn Iver Smith was"
        '''
        question = questions.split('\n')[0]
        question = re.sub(r'Question\d+: ', '', question)
        return question.strip()

    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            for q_id, q in enumerate(questions[:1]):
                question = self.split_question(q['Question'])
                try:
                    question_prefix = self.split_question(q['Question_prefix'])
                except KeyError:
                    question_prefix = self.split_question(q['Standard_Question'])
                # For single retrieval we only take the first question
                self.data.append({"context_id":context_id, "q_id":q_id, "question_prefix":question_prefix, "question":question, "answer":q['Answer'], "attribute":q['Property'][0]})
        logger.info(f"Dataset Preparation finished.")
    
    def length_check(self, tokenzier_length):
        if self.model_config.get("architectures") in [["Qwen2ForCausalLM"], ["InternLM3ForCausalLM"]]:
            # qwen and InternLM3 support 128k but without explicitly stating it in the config
            max_length = 131072
        elif "max_position_embeddings" in self.model_config.keys():
            max_length = self.model_config["max_position_embeddings"]
        elif "seq_length" in self.model_config.keys():
            max_length = self.model_config["seq_length"]
        else:
            return
        assert tokenzier_length < max_length, f"{tokenzier_length} excceed max length {max_length}"
        return

    def __len__(self):
        if self.data:
            return len(self.data)
        else:
            return len(self.full_data)

    def __getitem__(self, idx):
        datapoint = self.data[idx]
        context = self.contexts[datapoint['context_id']]
        question_prefix = datapoint['question_prefix']
        question_prefix = question_prefix[0].lower() + question_prefix[1:]
        question = datapoint['question']
        examples = datapoint.get('examples')
        if examples:
            prompt = fill_template(
            self.template,
            given_context=context,
            question=question,
            examples=examples,
        )
        else:
            prompt = fill_template(
                self.template,
                given_context=context,
                question_prefix=question_prefix, 
                question=question,
            )
        label = datapoint['answer']
        if isinstance(label, list):
            label = [str(l) for l in label]
        if self.tokenizer != 'gpt':
            tokenzier_length = len(self.tokenizer.tokenize(prompt))
            self.length_check(tokenzier_length)
        return datapoint['context_id'], datapoint['q_id'], question, prompt, label

class LongContextDataSetForIDK(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="IDK"):
        super().__init__(model_path, data_path, skill=skill)    
    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            questions = data['questions']['NLG']
            for q_id, q in enumerate(questions[:1]):
                question = self.split_question(q['Question'])
                answer_positions = q['position_in_list']
                positions_in_bios = q['position_in_bios']
                for ans_pos, pos_in_bios in zip(answer_positions, positions_in_bios):
                    bio = data['context'][ans_pos]
                    idk_bio = bio[:pos_in_bios[0]] + bio[pos_in_bios[1]:] # delete the answer
                    data['context'][ans_pos] = idk_bio
                # For single retrieval we only take the first question
                self.data.append({"context_id":context_id, "q_id":q_id, "question_prefix":"The answer is", "question":question, "answer":"not stated", "attribute":q['Property'][0]})
            self.contexts.append("".join(data['context'])) # append the modified data
        logger.info(f"Dataset Preparation finished.")

class LongContextDataSetForRank(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="Rank"):
        super().__init__(model_path, data_path, skill=skill)

class FewShotDataSetForRank(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="FewShotRank"):
        super().__init__(model_path, data_path, skill=skill)
    
    def get_few_shot_examples(self, few_shot_qs):
        demos = ""
        for i, q in enumerate(few_shot_qs):
            question = q['Question']
            rationale = q['Rationale'][0].lower() + q['Rationale'][1:] # Lower the first letter
            Question_prefix = q['Question_prefix'][0].lower() + q['Question_prefix'][1:]
            answer = q['Answer']
            demos += f"Question: {question}\nAnswer: Based on the provided context, {rationale} Therefore, {Question_prefix} {answer}.\n"
        return demos
    
    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            q, few_shot_qs = questions[0], questions[1:]
            examples = self.get_few_shot_examples(few_shot_qs)
            self.data.append({"context_id":context_id, "q_id":0, "question_prefix":q['Question_prefix'], "question":q['Question'], "examples": examples, "answer":q['Answer'], "attribute":q['Property']})
        logger.info(f"Dataset Preparation finished.")
    
    def __getitem__(self, idx):
        datapoint = self.data[idx]
        context = self.contexts[datapoint['context_id']]
        question_prefix = datapoint['question_prefix']
        question_prefix = question_prefix[0].lower() + question_prefix[1:]
        question = datapoint['question']
        examples = datapoint['examples']
        prompt = fill_template(
        self.template,
        given_context=context,
        question_prefix=question_prefix,
        question=question,
        examples=examples,
        )

        label = datapoint['answer']
        if isinstance(label, list):
            label = [str(l) for l in label]
        if self.tokenizer != 'gpt':
            tokenzier_length = len(self.tokenizer.tokenize(prompt))
            self.length_check(tokenzier_length)
        return datapoint['context_id'], datapoint['q_id'], question, prompt, label

class LongContextDataSetForCal(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="Calculation"):
        super().__init__(model_path, data_path, skill=skill)
    
    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            # for q_id, q in enumerate(questions):
            q = questions[0]
            self.data.append({"context_id":context_id, "q_id":0, "question_prefix":q['Question_prefix'], "question":q['Question'], "answer":q['Answer'], "attribute":q['Property'][0]})
        logger.info(f"Dataset Preparation finished.")

class FewShotDataSetForCal(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="FewShotCalculation"):
        super().__init__(model_path, data_path, skill=skill)
    
    def get_few_shot_examples(self, few_shot_qs):
        demos = ""
        for i, q in enumerate(few_shot_qs):
            question = q['Question']
            rationale = q['Rationale'][0].lower() + q['Rationale'][1:] # Lower the first letter
            Question_prefix = q['Question_prefix'][0].lower() + q['Question_prefix'][1:]
            answer = q['Answer']
            demos += f"Question: {question}\nAnswer: Based on the provided context, {rationale} Therefore, {Question_prefix} {answer}.\n"
        return demos
    
    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            q, few_shot_qs = questions[0], questions[1:]
            examples = self.get_few_shot_examples(few_shot_qs)
            self.data.append({"context_id":context_id, "q_id":0, "question_prefix":q['Question_prefix'], "question":q['Question'], "examples": examples, "answer":q['Answer'], "attribute":q['Property']})
        logger.info(f"Dataset Preparation finished.")

class LongContextDataSetForTwoDiff(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="TwoDiff"):
        super().__init__(model_path, data_path, skill=skill)
    
    def prepare_dataset(self):
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            for q_id, q in enumerate(questions[:1]):
                # For single retrieval we only take the first question
                self.data.append({"context_id":context_id, "q_id":q_id, "question_prefix":q['Question_prefix'], "question":q['Question'], "answer":q['Answer']['age_difference'], "attribute":q['Property']})
        logger.info(f"Dataset Preparation finished.")

class FewShotDataSetForTwoDiff(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="FewShotTwoDiff"):
        super().__init__(model_path, data_path, skill=skill)
    
    def get_few_shot_examples(self, few_shot_qs):
        demos = ""
        for i, q in enumerate(few_shot_qs):
            question = q['Question']
            rationale = q['Rationale']
            # Question_prefix = q['Question_prefix'][0].lower() + q['Question_prefix'][1:]
            answer_names = q['Answer']['ref_names']
            demos += f"Question: {question}\nAnswer: {rationale}\n"
        return demos
    
    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            q, few_shot_qs = questions[0], questions[1:]
            examples = self.get_few_shot_examples(few_shot_qs)
            self.data.append({"context_id":context_id, "q_id":0, "question_prefix":q['Question_prefix'], "question":q['Question'], "examples": examples, "answer":q['Answer']['age_difference'], "attribute":q['Property']})
        logger.info(f"Dataset Preparation finished.")


class LongContextDataSetForMultiQA(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="MultiQA"):
        self.model_path = model_path
        super().__init__(model_path, data_path, skill=skill)
    
    def prepare_prompt_template(self):
        messages = PROMPT_TEMPLATE[self.skill]
        logger.info(f"{self.skill} Prompt template applied: {messages}")
        if "mistral-nemo-7b" in self.model_path.lower():
            messages[2] = {"role": "assistant", "content": "Based on the provided context, the answers are:"}
        if self.tokenizer == 'gpt':
            return messages
        if "instruct" in self.model_path.lower():
            prompt_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
        else:
            logger.warning("No chat template find. Try to use template for base model")
            if self.skill + "ForBase" in PROMPT_TEMPLATE.keys():
                return PROMPT_TEMPLATE[self.skill + "ForBase"]
            else:
                assert False, "Current task did not support base model. Please turn to fewshot case"
        
        prompt_template = self.trim_string_until_target(prompt_template, messages[-1]["content"].strip())
        return prompt_template
    
    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            for q_id, q in enumerate(questions[:1]):
                question = q['Question']
                question_prefix = q['Question_prefix']
                # For single retrieval we only take the first question
                self.data.append({"context_id":context_id, "q_id":q_id, "question_prefix":question_prefix, "question":question, "answer":q['Answer'], "attribute":q['Property'][0]})
        logger.info(f"Dataset Preparation finished.")


class FewShotDataSet(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="FewShotSingleQA"):
        super().__init__(model_path, data_path, skill=skill)

    def get_two_shot_examples(self, questions):
        test_question_prefixs = questions[0]['Question_prefix']
        test_question = questions[0]['Question']
        test_answers = questions[0]['Answer'] 
        demos = ""
        for i, question in enumerate(questions[1:]):
            demo_question_prefix = question['Question_prefix'][0].lower() + question['Question_prefix'][1:]
            demo_question = question['Question']
            demo_answers = question['Answer'][0]
            demos += f"Question: {demo_question}\nAnswer: Based on the provided context, {demo_question_prefix} {demo_answers}.\n"
        return demos, test_question, test_question_prefixs, test_answers
        
    
    def get_two_shot_examples_from_multi_retrieval(self, questions):
        question_prefixs = questions[0]['Question_prefix']
        questions = questions[0]['Question']
        answers = questions[0]['Answer']
        splited_question = [line.split(': ', 1)[1] for line in questions.split('\n')]
        splited_question_prefix = [line.split(': ', 1)[1] for line in question_prefixs.split('\n')]
        assert len(splited_question) == 3, "should have 3 retrieval dataset for single 2-shot"
        demos = ""
        for i, (std_question, question, answer) in enumerate(zip(splited_question_prefix, splited_question, answers)):
            demos += f"Question{i}: {question}\nAnswer: Based on the provided context, {std_question.strip()} {answer}.\n"
            if i+1 == 2:
                # Only take the first two questions as examples:
                break
        return demos, splited_question[2], splited_question_prefix[2], answers[2:]
    
    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            if len(questions) > 1:
                demos, question, question_prefix, answer = self.get_two_shot_examples(questions)
            else:
                #legacy
                demos, question, question_prefix, answer = self.get_two_shot_examples_from_multi_retrieval(questions)
            self.data.append({"context_id":context_id, "q_id":0, "question_prefix":question_prefix, "question":question, "answer":answer, "attribute":questions[0]['Property'], "demos": demos})
        logger.info(f"Dataset Preparation finished.")
    
    def __getitem__(self, idx):
        datapoint = self.data[idx]
        context = self.contexts[datapoint['context_id']]
        question_prefix = datapoint['question_prefix']
        question = datapoint['question']
        demos = datapoint['demos']
        prompt = fill_template(
            self.template,
            given_context=context,
            question=question,
            examples=demos,
            question_prefix=question_prefix
        )
        label = datapoint['answer']
        return datapoint['context_id'], datapoint['q_id'], question, prompt, label

class FewShotDataSetForMultiQA(FewShotDataSet):
    def __init__(self, model_path, data_path, skill="FewShotMultiQA"):
        super().__init__(model_path, data_path, skill=skill) 

class LongContextDataSetForMultihop(LongContextDataSet):
    def __init__(self, model_path, data_path, skill="Multihop"):
        super().__init__(model_path, data_path, skill=skill)
    
    def prepare_dataset(self):
        # logger.info(f"Start preparing on {task_name} task.")
        self.data = []
        self.contexts = []
        for context_id, data in tqdm(enumerate(self.full_data)):
            self.contexts.append("".join(data['context']))
            questions = data['questions']['NLG']
            for q_id, q in enumerate(questions[:1]):
                question = q['Question']
                try:
                    question_prefix = self.split_question(q['Question_prefix'])
                except KeyError:
                    question_prefix = self.split_question(q['Standard_Question'])
                # For single retrieval we only take the first question
                self.data.append({"context_id":context_id, "q_id":q_id, "question_prefix":question_prefix, "question":question, "answer":q['Answer'], "attribute":q['Property'][0]})
        logger.info(f"Dataset Preparation finished.")

class FewShotDataSetForMultihop(FewShotDataSet):
    def __init__(self, model_path, data_path, skill="FewShotMultihop"):
        super().__init__(model_path, data_path, skill=skill) 
