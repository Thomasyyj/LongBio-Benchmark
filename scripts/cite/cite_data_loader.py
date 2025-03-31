import numpy as np
from torch.utils.data import Dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from scripts.data_loader_prompts import CITE_PROMPT_TEMPLATE
from scripts.data_loader import LongContextDataSet

class LongContextCiteDataSetFactory:
    def __new__(cls, args, **kwargs):
        cite_tasks_map = {
            "SingleQA": CiteDataSet,
            "MultiQA": CiteDataSetForMultiQA,
            "FewshotSingleQA": CiteDataSetForSingleFewShotQA,
            "FewshotMultiQA": CiteDataSetForMultiFewShotQA
        }
        if args.apply_fewshot:
            if args.n_questions > 1:
                skill = "FewshotMultiQA"
            else:
                skill = "FewshotSingleQA"
        else:
            if args.n_questions > 1:
                skill = "MultiQA"
            else:
                skill = "SingleQA"
        return cite_tasks_map[skill](**kwargs)


class CiteDataSet(LongContextDataSet):
    def __init__(self, model_path, data_path, context_length, skill="SingleQA"):
        self.skill = skill
        super().__init__(model_path, data_path, skill=self.skill)
        self.context_length = context_length * 1024 - 200 # reserve place for output
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    def prepare_prompt_template(self):
        messages = CITE_PROMPT_TEMPLATE[self.skill]
        logger.info(f"{self.skill} Prompt template applied: {messages}")

        try:
            prompt_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
        except ValueError as e:
            logger.warning("No chat template find. Try to use template for base model")
            assert False, "CITE task does not support base model. Please refer to few-shot CITE task"
        
        prompt_template = self.trim_string_until_target(prompt_template, messages[-1]["content"].strip())
        return prompt_template
    
    @staticmethod
    def truncate(list1, answer_position):
        """
        Removes the last element from list1 whose index is not in answer_position.
        Updates answer_position to reflect the indices of the safe elements in the modified list1.

        :param list1: The original list.
        :param answer_position: A list of indices of safe elements in list1.
        :return: A tuple containing the modified list1 and the updated answer_position.
        """
        # Find the last element in list1 whose index is not in answer_position
        idx_to_pop = None
        for idx in range(len(list1) - 1, -1, -1):
            if idx not in answer_position:
                idx_to_pop = idx
                break

        # If all elements are safe, return the original lists
        if idx_to_pop is None:
            return list1, answer_position

        # Remove the element from list1
        list1.pop(idx_to_pop)

        # Update answer_position to reflect the new indices in list1
        updated_answer_position = []
        for idx in answer_position:
            if idx < idx_to_pop:
                # Indices before the removed element remain the same
                updated_answer_position.append(idx)
            elif idx > idx_to_pop:
                # Indices after the removed element shift left by 1
                updated_answer_position.append(idx - 1)
            # If idx == idx_to_pop, it should not happen since we don't remove safe elements

        return list1, updated_answer_position
    
    def prepare_context(self, context_list):
        out_context = ""
        for b_i, bio in enumerate(context_list):
            out_context += f"\nBio [{b_i}]: " + bio[20:-29]
        return out_context

    def prepare_dataset(self):
        self.data = []
        self.contexts = []
        for context_id, data in enumerate(self.full_data):
            questions = data['questions']['NLG']
            answer_positions = questions[0]['position_in_list'] # Choose the first question. The rest are examples

            context_list = data['context']
            context = self.prepare_context(context_list)
            context_length = len(self.tokenizer.tokenize(context))
            while context_length > self.context_length:
                logger.info(f"{context_length} Exceeds Maximum Length {self.context_length}. Truncating the bios..")
                context_list, answer_positions = self.truncate(context_list, answer_positions) # update both context and answer positions after truncating
                context = self.prepare_context(context_list)
                context_length = len(self.tokenizer.tokenize(context))
            
            self.contexts.append(context)
            q = questions[0] # Only take the first as question, the rest are few shot examples
            self.data.append({"context_id":context_id, "q_id":0, "question_prefix":q['Question_prefix'], "question":q['Question'], "answer":q['Answer'], "attribute":q['Property'], "citation":answer_positions})


        logger.info(f"Dataset Preparation finished.")

    def __getitem__(self, idx):
        datapoint = self.data[idx]
        context = self.contexts[datapoint['context_id']]
        question_prefix = datapoint['question_prefix']
        question_prefix = question_prefix[0].lower() + question_prefix[1:]
        question = datapoint['question']
        prompt = self.template.format(
            given_context=context,
            question_prefix=question_prefix, 
            question=question,
        )
        label = datapoint['answer']
        citation = [f'[{c}]' for c in datapoint['citation']]
        return datapoint['context_id'], datapoint['q_id'], question, prompt, label, citation

class CiteDataSetForMultiQA(CiteDataSet):
    def __init__(self, model_path, data_path, context_length, skill="MultiQA"):
        super().__init__(model_path, data_path, context_length, skill=skill)

class CiteDataSetForSingleFewShotQA(CiteDataSet):
    def __init__(self, model_path, data_path, context_length, skill="SingleFewShotQA"):
        super().__init__(model_path, data_path, context_length, skill=skill) 
    
    def prepare_prompt_template(self):
        messages = CITE_PROMPT_TEMPLATE[self.skill]
        logger.info(f"{self.skill} Prompt template applied: {messages}")

        try:
            prompt_template = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, continue_final_message=True)
            prompt_template = self.trim_string_until_target(prompt_template, messages[-1]["content"].strip())
        except ValueError as e:
            logger.info("No chat template found. Turn to template for base model")
            if "single" in self.skill.lower():
                prompt_template = CITE_PROMPT_TEMPLATE["SingleFewShotQAForBase"]
            elif "multi" in self.skill.lower():
                prompt_template = CITE_PROMPT_TEMPLATE["MultiFewShotQAForBase"]
            else:
                assert False, f"Unsupported skill {self.skill}"
        return prompt_template
    
    def get_two_shot_examples(self, questions:list, citations: list[list[int]]):
        demos = ""
        # citations = citations[0] # single retrieval case
        for i, (q, citation) in enumerate(zip(questions[1:], citations[1:])):
            # Examples start from the second data
            demos += f"Question: {q['Question']}\nAnswer: Based on the provided context, {q['Question_prefix']} {q['Answer']} [{citation}].\n"
            if i+1 == 2:
                # Only take the first two questions as examples:
                break
        return demos, questions[0]['Question'], questions[0]['Question_prefix'], questions[0]['Answer'], citations[0]
            

    def prepare_dataset(self):
        self.data = []
        self.contexts = []
        for context_id, data in enumerate(self.full_data):
            questions = data['questions']['NLG']
            n_q = len(questions)
            n_citation = len(questions[0]['position_in_list'])
            answer_positions = [position for q in questions for position in q['position_in_list']] # shape (n_q, n_citation) -> (n_q * n_citation)
            context_list = data['context']
            context = self.prepare_context(context_list)
            context_length = len(self.tokenizer.tokenize(context))
            while context_length > self.context_length:
                logger.info(f"{context_length} Exceeds Maximum Length {self.context_length}. Truncating the bios..")
                context_list, answer_positions = self.truncate(context_list, answer_positions) # update both context and answer positions after truncating
                context = self.prepare_context(context_list)
                context_length = len(self.tokenizer.tokenize(context))
            answer_positions = np.array(answer_positions).reshape(n_q, n_citation).tolist()
            self.contexts.append(context)
                
            demos, question, question_prefix, answer, citation = self.get_two_shot_examples(questions, answer_positions)
            self.data.append({"context_id":context_id, "q_id":0, "question_prefix":question_prefix, "question":question, "answer":answer, "citation":citation, "demos": demos})

        logger.info(f"Dataset Preparation finished.")

    def __getitem__(self, idx):
        datapoint = self.data[idx]
        context = self.contexts[datapoint['context_id']]
        question_prefix = datapoint['question_prefix']
        question = datapoint['question']
        demos = datapoint['demos']
        if "{question_prefix}" in self.template:
            question_prefix = question_prefix[0].lower() + question_prefix[1:]
            prompt = self.template.format(
                given_context=context,
                question=question,
                question_prefix=question_prefix,
                examples=demos
            )
        else:
            prompt = self.template.format(
                given_context=context,
                question=question,
                examples=demos
            )
        label = datapoint['answer']
        if isinstance(datapoint['citation'], int):
            citation = [f"[{datapoint['citation']}]"]
        else:
            citation = [f'[{c}]' for c in datapoint['citation']]
        return datapoint['context_id'], datapoint['q_id'], question, prompt, label, citation

class CiteDataSetForMultiFewShotQA(CiteDataSetForSingleFewShotQA):
    def __init__(self, model_path, data_path, context_length, skill="MultiFewShotQA"):
        super().__init__(model_path, data_path, context_length, skill=skill) 
    
    def get_two_shot_examples(self, questions:list, citations: list[list[int]]):
        demos = ""
        for i, (q, citation) in enumerate(zip(questions[1:], citations[1:])):
            # Examples start from the second data
            demo_answers = ""
            for prefix_i, prefix in enumerate(q['Question_prefix'].split('\n')):
                prefix = prefix.split(': ')[-1].strip()
                # e.g Avery Nathaniel Norman was born in Isabela [4].
                demo_answers += f"{prefix} {q['Answer'][prefix_i]} [{citation[prefix_i]}]. "
            demos += f"Examples{i+1}: {q['Question']}\nAnswer: Based on the provided context, {demo_answers}\n"
        return demos, questions[0]['Question'], questions[0]['Question_prefix'], questions[0]['Answer'], citations[0]
