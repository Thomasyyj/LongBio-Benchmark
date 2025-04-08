import os
import argparse
import random
import tiktoken

from typing import Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from loguru import logger
import json

from bios import Bio, reasoning_question_template
from utils import sample_boolen, join_names, generate_random_date, read_list_from_txt, compare_size
from __init__ import DATA_UTIL_PATH

class BioQABenchmarkGenerator:
    
    def __init__(
        self,
        length: float,
        num_retrieval: int, # the number of information piece to retrieve
        position: Tuple, # the position of the information piece to retrieve in the entire conext,
        density: float, # the density of the distractors
        apply_fewshot: bool,
        reasoning_skill: str = None,
        num_bios: int = 10000, 
        attribute_dir: str = os.path.join(DATA_UTIL_PATH, "attributes"),
        tokenizer_path: str = None,
    ) -> None:
        self.apply_fewshot = apply_fewshot
        self.num_retrieval = num_retrieval + 2 * num_retrieval * self.apply_fewshot # we use 2 shot by default
        self.named_length = length
        # assert self.named_length in [4, 8, 16, 32, 64, 128], f"support length [4, 8, 16, 32, 64, 128], {length} found"

        if 'gpt' in tokenizer_path:
            self.length = length*1000 - 150 - (200 * self.num_retrieval) # For gpt
            logger.info(f"For gpt, max length set to {length}*1000 - 200 - (200 * {self.num_retrieval}) = {self.length}")
        else:
            self.length = length*1024 - 150 - (200 * self.num_retrieval) # reserve tokens for prompts and question bios
            logger.info(f"Max length set to {length}*1024 - 200 - (200 * {self.num_retrieval}) = {self.length}")

        self.position = position
        self.density = density
        self.reasoning_skill = reasoning_skill
        self.num_bios = num_bios
        self.attribute_dir = attribute_dir
        self.tokenizer_path = tokenizer_path
        if 'gpt' in tokenizer_path:
            tokenizer_path = tokenizer_path.split('/')[-1]
            self.tokenizer = tiktoken.encoding_for_model(tokenizer_path)
            self.tokenizer.tokenize = self.tokenizer.encode
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True) if self.tokenizer_path else None
        
        # init
        self.attribute_pool = self.get_all_attributes(self.attribute_dir)
        self.birthdate_pool = ("1950-01-01", "2001-12-31")
        self.random_seed = 2024

        self.q_attributes = list(
            filter(lambda x: x not in ["first_name", "middle_name", "last_name", "gender"], self.attribute_pool.keys()))
        self.q_attributes += ["birthdate"]
        logger.info(f"Preparing seed {self.num_bios} bios")
        self.bios = []
        while len(self.bios) < self.num_bios:
            new_bio = Bio(**self.sample_attributes())
            if new_bio not in self.bios:
                self.bios.append(new_bio)
        logger.info(f"Seed bios preparation finished")
        self.suffix = new_bio.suffix # We need to remove this suffix at the end of the context
        
        self.dataset = []
    
    def remove_suffix(self, text):
        if text.endswith(self.suffix):
            return text[: -len(self.suffix)].rstrip()
        return text
        
    def sample_attributes(self):
        sampled_attr = {}
        for k, v in self.attribute_pool.items():
            sampled_attr[k] = random.choice(v)
        birthdate = generate_random_date(self.birthdate_pool[0], self.birthdate_pool[1])
        sampled_attr['birthdate'] = birthdate.strftime("%Y-%m-%d")
        sampled_attr['age'] = 2024 - birthdate.year
        return sampled_attr
    
    @staticmethod
    def get_all_attributes(attribute_dir):
        attribute_pool = {}
        for attribute_file in os.listdir(attribute_dir):
            attribute = attribute_file.split(".")[0]
            logger.info(attribute_file, attribute)
            attribute_pool[attribute] = read_list_from_txt(os.path.join(attribute_dir, attribute_file))
        return attribute_pool
        
    def single_retrieval(
        self,
        use_paraphrase: bool = False,
        use_pronoun: bool = False,
        random_attr: bool = False,
    ):
        assert self.num_retrieval == 1, "This function is only for single retrieval"
        assert self.apply_fewshot==False, "Not support few-shot for single retrieval"
        # Bio1 + ... + BioN + Q + A
        # 
        
        # 1. sample one Bio and one of it's attribution -> question
        bio_q, attr_q = random.choice(self.bios), random.choice(self.q_attributes)
        s_q, q = bio_q.to_question(attr_q)
        answer = bio_q.__getattribute__(attr_q)
        
        # 2. construct context (context does not contain sampled bio) - given density
        context, context_lengths = [], []
        context_length = 0
        n_bios = 0
        rng = random.Random(self.random_seed) 
        while True: # check the length before concating the new context
            b = random.choice(self.bios)
            if b != bio_q:
                desc = b.to_description(
                    rng,
                    exclude_attributions=[attr_q] if not sample_boolen(self.density, rng) else [],
                    use_pronoun=use_pronoun,
                    use_paraphrase=use_paraphrase,
                    random_attr=random_attr
                )["description"]

                increased_length = len(self.tokenizer.tokenize(" "+desc.strip())) if self.tokenizer is not None else len(desc.split())

                if context_length + increased_length > self.length:
                    if 'llama' in self.tokenizer_path:
                        context_length = context_length + 1 # add the last 'Ġ'
                        context_lengths[-1] = context_lengths[-1] + 1
                    break 

                context.append(desc)
                context_length += increased_length
                n_bios += 1
                context_lengths.append(context_length)
        # Note that here the tokenzier is not exactly the same
        # 3. insert the sampeld bio in the context at the right position
        possible_positions = [i for i, l in enumerate(context_lengths) if context_length * self.position[0] <= l <= context_length * self.position[1]]
        pos = random.choice(possible_positions)
        description_object_q = bio_q.to_description(rng, use_pronoun=use_pronoun, use_paraphrase=use_paraphrase, random_attr=random_attr, freeze_attr=attr_q)
        desc = description_object_q["description"]
        context.insert(pos, desc)
        context_length = len(self.tokenizer.tokenize("".join(context))) if self.tokenizer is not None else len("".join(context))
        if 'gpt' in self.tokenizer_path:
            assert self.named_length * 1000 - 100 > context_length, f"{context_length} Excced max length"
        else:
            assert self.named_length * 1024 - 100 > context_length, f"{context_length} Excced max length"
        # context_length += len(self.tokenizer.tokenize(" "+description_object_q["description"].strip())) if self.tokenizer is not None else len(description_object_q["description"].split())
        q_ans_position = description_object_q[attr_q]['position']

        context[-1] = self.remove_suffix(context[-1]) # remove the suffix at the end
        # generate a single retrieval test data point
        data_point = {
            "context": context,
            "token_length": context_length,
            "questions":{
                "NLG": [
                    {
                        'Question_prefix' : s_q,
                        'Question' : q,
                        'Answer': answer,
                        'Property': attr_q,
                        'position_in_list': [pos],
                        'position_in_bios': [q_ans_position],
                        'reference': context[pos][q_ans_position[0]:q_ans_position[1]]
                    }
                ]
            }
        }

        return data_point
    
    def single_retrieval_paraphrase(self):
        return self.single_retrieval(use_pronoun=False, use_paraphrase=True)
    
    def single_retrieval_pronoun(self):
        return self.single_retrieval(use_pronoun=True, use_paraphrase=True)
    
    def single_retrieval_rnidk(self):
        return self.single_retrieval(random_attr=True)
    
    def get_context_questions_for_multiple_retrieval(self, attr_q, bio_qs, qs, s_qs, use_pronoun, use_paraphrase, use_age):
        context, context_length_list = [], []
        context_length = 0
        n_bios = 0
        rng = random.Random(self.random_seed)
        while True:
            b = random.choice(self.bios)
            if b not in bio_qs:
                desc = b.to_description(
                    rng,
                    exclude_attributions=[attr_q] if not sample_boolen(self.density, rng) else [],
                    use_pronoun=use_pronoun,
                    use_paraphrase=use_paraphrase,
                    use_age=use_age
                )["description"]
                increased_length = len(self.tokenizer.tokenize(" "+desc.strip())) if self.tokenizer is not None else len(desc.split())

                if context_length + increased_length > self.length:
                    if 'llama' in self.tokenizer_path:
                        context_length = context_length + 1 # add the last 'Ġ'
                        context_length_list[-1] = context_length_list[-1] + 1
                    break 
                
                context.append(desc)
                context_length += increased_length
                n_bios += 1
                context_length_list.append(context_length)

        possible_positions = [i for i, l in enumerate(context_length_list) if context_length * self.position[0] <= l <= context_length * self.position[1]]
        pos = random.choices(possible_positions, k=len(bio_qs))
        pos = sorted(pos)
        offset = 0
        
        question_list = []
        for idx, (p, bio_q) in enumerate(zip(pos, bio_qs)):
            description_object_q = bio_q.to_description(rng, use_pronoun=use_pronoun, use_paraphrase=use_paraphrase,  use_age=use_age)
            desc = description_object_q["description"]
            context.insert(p + offset, desc)
            offset += 1
            context_length += len(self.tokenizer.tokenize(" "+desc.strip())) if self.tokenizer is not None else len(desc.split())
            q_ans_position = description_object_q[attr_q]['position']
            question_list.append(
                {   
                    'Question_prefix': s_qs[idx],
                    'Question' : qs[idx],
                    'Answer': bio_q.__getattribute__(attr_q),
                    'Property': attr_q,
                    'position_in_list': [p + offset - 1],
                    'position_in_bios': [q_ans_position],
                    'reference': context[p + offset - 1][q_ans_position[0]:q_ans_position[1]]
                }
            )
            
        context_length = len(self.tokenizer.tokenize("".join(context))) if self.tokenizer is not None else len(" ".join(context))
        if 'gpt' in self.tokenizer_path:
            assert self.named_length * 1000 - 100 > context_length, f"{context_length} Excced max length"
        else:
            assert self.named_length * 1024 - 100 > context_length, f"{context_length} Excced max length"
        return context, question_list, context_length
    
    def get_context_questions_for_multihop_retrieval(self, attr_q, bio_qs, qs, s_qs, use_pronoun, use_paraphrase, use_age):
        context, context_length_list = [], []
        context_length = 0
        n_bios = 0
        rng = random.Random(self.random_seed)
        while True:
            b = random.choice(self.bios)
            if b not in bio_qs:
                desc = b.to_description(
                    rng,
                    exclude_attributions=[attr_q] if not sample_boolen(self.density, rng) else [],
                    use_pronoun=use_pronoun,
                    use_paraphrase=use_paraphrase,
                    use_age=use_age
                )["description"]
                increased_length = len(self.tokenizer.tokenize(" "+desc.strip())) if self.tokenizer is not None else len(desc.split())

                if context_length + increased_length > self.length:
                    if 'llama' in self.tokenizer_path:
                        context_length = context_length + 1 # add the last 'Ġ'
                        context_length_list[-1] = context_length_list[-1] + 1
                    break 
                
                context.append(desc)
                context_length += increased_length
                n_bios += 1
                context_length_list.append(context_length)

        possible_positions = [i for i, l in enumerate(context_length_list) if context_length * self.position[0] <= l <= context_length * self.position[1]]
        pos = random.choices(possible_positions, k=len(bio_qs))
        pos = sorted(pos)
        offset = 0
        
        question_list = []
        for idx, (p, bio_q) in enumerate(zip(pos, bio_qs)):
            if self.apply_fewshot:
                n_retrieval = len(bio_qs)//3
                # For each question (two shots), change the second question
                if idx % n_retrieval != 0:
                    description_object_q = bio_q.to_description(rng, use_pronoun=use_pronoun, use_paraphrase=use_paraphrase,  use_age=use_age, ref_bio_name_and_attr=(bio_qs[idx-1].full_names, attr_q))
                else:
                    description_object_q = bio_q.to_description(rng, use_pronoun=use_pronoun, use_paraphrase=use_paraphrase,  use_age=use_age)
                    answer = bio_q.__getattribute__(attr_q)
            else:
                # From the second question, change the context in to "The attribute is the same as the previous one"
                if idx > 1:
                    description_object_q = bio_q.to_description(rng, use_pronoun=use_pronoun, use_paraphrase=use_paraphrase,  use_age=use_age, ref_bio_name=bio_qs[idx-1].full_names)
                else:
                    description_object_q = bio_q.to_description(rng, use_pronoun=use_pronoun, use_paraphrase=use_paraphrase,  use_age=use_age)
                    answer = bio_q.__getattribute__(attr_q)
            desc = description_object_q["description"]
            context.insert(p + offset, desc)
            offset += 1
            context_length += len(self.tokenizer.tokenize(" "+desc.strip())) if self.tokenizer is not None else len(desc.split())
            q_ans_position = description_object_q[attr_q]['position']
            question_list.append(
                {   
                    'Question_prefix': s_qs[idx],
                    'Question' : qs[idx],
                    'Answer': answer,
                    'Property': attr_q,
                    'position_in_list': [p + offset - 1],
                    'position_in_bios': [q_ans_position],
                    'reference': context[p + offset - 1][q_ans_position[0]:q_ans_position[1]]
                }
            )
            
        context_length = len(self.tokenizer.tokenize("".join(context))) if self.tokenizer is not None else len(" ".join(context))
        if 'gpt' in self.tokenizer_path:
            assert self.named_length * 1000 - 100 > context_length, f"{context_length} Excced max length"
        else:
            assert self.named_length * 1024 - 100 > context_length, f"{context_length} Excced max length"
        return context, question_list, context_length

    def prepare_multi_retrieval_questions(self, input_question_lists, subtask):
        rationale = ""
        if len(input_question_lists) == 1:
            question = input_question_lists[0]['Question']
            questions_prefix = input_question_lists[0]['Question_prefix']
            answer = [q['Answer'] for q in input_question_lists]
        else:
            if subtask == 'retrieval':
                question = "\n".join([f"Question{i+1}: " + q["Question"] for i, q in enumerate(input_question_lists)])
                questions_prefix = "\n".join([f"Question{i+1}: " + q["Question_prefix"] for i, q in enumerate(input_question_lists)])
                answer = [q['Answer'] for q in input_question_lists]
            elif subtask == 'multihop':
                # Only take the last question as the starting of multi-hop
                question = input_question_lists[-1]["Question"]
                questions_prefix = input_question_lists[-1]['Question_prefix']
                answer = input_question_lists[-1]["Answer"]
                for q in reversed(input_question_lists): 
                    rationale += q['reference']
            else:
                assert False, "Should not go here."

        return [question], [rationale], [questions_prefix], [answer]

    def multiple_retrieval(self, use_pronoun: bool = False, use_paraphrase: bool = False, use_age: bool = False, use_multihop=False):
        # Bio1 + ... + BioN + Q1 + Q2 + ... + QN + A1 + A2 + ... + AN, all questions is about one same attributes, e.g., the birthdate
        assert self.num_retrieval > 1, "This function is only for multiple retrieval"
        # 1. sample the bios and the attributes
        bio_qs, attr_q = random.choices(self.bios, k=self.num_retrieval), random.choices(self.q_attributes, k=1)[0]
        s_qs = [bio_q.to_question(attr_q)[0] for bio_q in bio_qs]
        qs = [bio_q.to_question(attr_q)[1] for bio_q in bio_qs]
        
        # 2. construct context length 
        if use_multihop:
            context, question_list, context_length = self.get_context_questions_for_multihop_retrieval(attr_q, bio_qs, qs, s_qs, use_pronoun, use_paraphrase, use_age)
            mode = 'multihop'
        else:
            context, question_list, context_length = self.get_context_questions_for_multiple_retrieval(attr_q, bio_qs, qs, s_qs, use_pronoun, use_paraphrase, use_age)
            mode = 'retrieval'

        if self.apply_fewshot:
            questions, rationales, questions_prefix, answers = self.prepare_few_shot_questions(question_list, mode, self.prepare_multi_retrieval_questions)
            num_retrieval_per_demo = self.num_retrieval//3 # Assume 2 shot
        else:
            questions, rationales, questions_prefix, answers = self.prepare_multi_retrieval_questions(question_list, mode)
            num_retrieval_per_demo = self.num_retrieval

        qa_data = []
        for idx in range(len(questions)):
            # One demo needs num_retrieval_per_demo number of reference
            question_list_per_demo = question_list[idx*num_retrieval_per_demo:(idx+1)*num_retrieval_per_demo]
            qa_data.append(
                {
                    "Question_prefix": questions_prefix[idx],
                    "Question": questions[idx],
                    "Rationale": rationales[idx],
                    "Answer": answers[idx],
                    "Property": [q['Property'] for q in question_list_per_demo],
                    "position_in_list": [q["position_in_list"][0] for q in question_list_per_demo],
                    "position_in_bios": [q["position_in_bios"][0] for q in question_list_per_demo],
                    "reference": [q["reference"] for q in question_list_per_demo]
                }
            ) 

        data_point = {
            "context": context,
            "token_length": context_length,
            "questions":{
                "NLG": qa_data
            }
        }
        
        return data_point

    def multiple_retrieval_paraphrase(self):
        return self.multiple_retrieval(use_pronoun=False, use_paraphrase=True)
    
    def multiple_retrieval_pronoun(self):
        return self.multiple_retrieval(use_pronoun=True, use_paraphrase=True)
    
    def prepare_math_reasoning_questions(self, bio_qs, subtask):
        assert subtask in ['rank', 'calculation', 'twodiff'], f"Undefined math reasoning subtask {subtask}"
        # reasoning_question_template
        if subtask == "rank":
            answer_dict = {bio_q.full_names:bio_q.__getattribute__("age") for bio_q in bio_qs}
            people_names = join_names(list(answer_dict.keys()))
            rationale = ""
            for name, age in answer_dict.items():
                rationale += f"The age of {name} is {age}. "
            question = reasoning_question_template["rank"].format(people_names=people_names)
            questions_prefix = f"The names list after ranking the age of {people_names} is"

            sorted_people = sorted(answer_dict.items(), key=lambda x: x[1])
            # Extract the sorted ages and corresponding names
            sorted_ages = [age for _, age in sorted_people]
            answer = [name for name, _ in sorted_people]
            sorted_names = [f"the age of {name}" for name in answer]
            rationale += ' < '.join(map(str, sorted_ages)) + '. '
            rationale += "Therefore, " + ' < '.join(map(str, sorted_names)) + '. '
        elif subtask == "calculation":
            answer_dict = {bio_q.full_names:bio_q.__getattribute__("age") for bio_q in bio_qs}
            people_names = join_names([bio_q.full_names for bio_q in bio_qs])
            rationale = ""
            for name, age in answer_dict.items():
                rationale += f"The age of {name} is {age}. "
            question = reasoning_question_template["calculation"].format(people_names=people_names)
            ages_list = list(answer_dict.values())
            questions_prefix = f"The age difference is"
            answer = abs(ages_list[1] - ages_list[0])
            rationale += f"|{ages_list[1]} - {ages_list[0]}| = {abs(ages_list[1] - ages_list[0])}. "
        else:
            # subtask == 'twodiff'
            # For twodiff, the answer is a dict
            names_list = [bio_q.full_names for bio_q in bio_qs]
            people_names = join_names(names_list)
            ages_list = [bio_q.__getattribute__("age") for bio_q in bio_qs]
            age_difference = abs(ages_list[1] - ages_list[0])

            rationale = f"To answer this question, we need to find two people in the context whose age difference is {age_difference}. "
            # for name, age in zip(names_list, ages_list):
            rationale += f"The age of {names_list[0]} is {ages_list[0]}. "
            if ages_list[0] - age_difference <= 0 or age_difference==0:
                rationale += f"Therefore, the age of the second person should be {ages_list[0] + age_difference}. "
            else:
                rationale += f"Therefore, the age of the second person should be {ages_list[0] + age_difference} or {ages_list[0] - age_difference}. "
            rationale += f"After searching from the context, the age of {names_list[1]} is {ages_list[1]}. "
            rationale += f"Therefore, {people_names} satisfy the question. "
            age_difference = abs(ages_list[1] - ages_list[0])
            question = reasoning_question_template["twodiff"].format(age_difference=age_difference)
            questions_prefix = f"The two people identified are"
            answer = {
                "age_difference": age_difference,
                "ref_names": names_list
            }
        # Ouput the list in order to have the same format as func prepare_few_shot_questions
        return [question], [rationale], [questions_prefix], [answer]
    
    def prepare_few_shot_questions(self, list_before_separation, subtask, prepare_func):
        num_retrieval_per_demo = len(list_before_separation)//3 # We use two shots by default

        questions = []
        rationales = []
        questions_prefix = []
        answers = []
        for i in range(0, len(list_before_separation), num_retrieval_per_demo):
            list_after_separation = list_before_separation[i:i+num_retrieval_per_demo]
            question, rationale, standard_question, answer = prepare_func(list_after_separation, subtask)
            questions.append(question[0])
            rationales.append(rationale[0])
            questions_prefix.append(standard_question[0])
            answers.append(answer[0])
        
        return questions, rationales, questions_prefix, answers

    def math_reasoning_retrieval(self, subtask, use_pronoun=False, use_paraphrase=False, use_age=True):
        '''
        Bio with reasoning retrieval
        rank: what is the rank of age between n people?
        calculation: what is the age difference between two people
        twodiff: What are n people whose ages are summed to be 20*n
        '''
         # 1. sample the bios and the attributes
         # ensure every age is different
        bio_age_list = [0 for _ in range(self.num_retrieval)]
        while len(bio_age_list) != len(set(bio_age_list)):
            bio_qs = random.choices(self.bios, k=self.num_retrieval)
            bio_age_list = [bio_q.__getattribute__("age") for bio_q in bio_qs]
        attr_q = "age" if use_age else "birthdate"
        s_qs = [bio_q.to_question(attr_q)[0] for bio_q in bio_qs]
        qs = [bio_q.to_question(attr_q)[1] for bio_q in bio_qs]
        # 2. construct context length (context does not contain bio_qs)
        context, question_list, context_length = self.get_context_questions_for_multiple_retrieval(attr_q, bio_qs, qs, s_qs, use_pronoun, use_paraphrase, use_age) 
        if self.apply_fewshot:
            questions, rationales, questions_prefix, answers = self.prepare_few_shot_questions(bio_qs, subtask, self.prepare_math_reasoning_questions)
            num_retrieval_per_demo = self.num_retrieval//3 # We use 2 shots by default, so the num retrieveal for each question is num_retrieval//3
        else:
            questions, rationales, questions_prefix, answers = self.prepare_math_reasoning_questions(bio_qs, subtask)
            num_retrieval_per_demo = self.num_retrieval
        qa_data = []
        for idx in range(len(questions)):
            # One demo needs num_retrieval_per_demo number of reference
            question_list_per_demo = question_list[idx*num_retrieval_per_demo:(idx+1)*num_retrieval_per_demo]
            qa_data.append(
                {
                    "Question_prefix": questions_prefix[idx],
                    "Question": questions[idx],
                    "Rationale": rationales[idx],
                    "Answer": answers[idx],
                    "Property": [q['Property'] for q in question_list_per_demo],
                    "position_in_list": [q["position_in_list"][0] for q in question_list_per_demo],
                    "position_in_bios": [q["position_in_bios"][0] for q in question_list_per_demo],
                    "reference": [q["reference"] for q in question_list_per_demo]
                }
            )
        
        data_point = {
            "context": context,
            "token_length": context_length,
            "questions":{
                "NLG": qa_data
            }
        }
        return data_point
    
    def math_reasoning_rank(self):
        if self.apply_fewshot:
            assert self.num_retrieval/2>=3, "This function only supports more than 2 retrieval"
        else:
            assert self.num_retrieval>=2, "This function only supports more than 2 retrieval"
        return self.math_reasoning_retrieval(subtask="rank")
    
    def math_reasoning_calulation(self):
        if self.apply_fewshot:
            assert self.num_retrieval/2==3, "This function only supports 2 retrieval"
        else:
            assert self.num_retrieval==2, "This function only supports 2 retrieval"
        return self.math_reasoning_retrieval(subtask="calculation")

    def math_reasoning_twodiff(self):
        if self.apply_fewshot:
            assert self.num_retrieval/2==3, "This function only supports 2 retrieval"
            return self.math_reasoning_retrieval(subtask="twodiff")
        else:
            assert self.num_retrieval==2, "This function only supports 2 retrieval"
            return self.math_reasoning_retrieval(subtask="twodiff")
    
    def multihop_reasoning(self):
        return self.multiple_retrieval(use_multihop=True) 

    def generate(self, num_data):
        if self.num_retrieval == 1:
            if self.reasoning_skill == 'standard':
                logger.info("Generating Single Retrieval Benchmark with standard baseline")
                gen_func = self.single_retrieval
            elif self.reasoning_skill == 'paraphrase':
                logger.info("Generating Single Retrieval Benchmark with praphrased templates")
                gen_func = self.single_retrieval_paraphrase
            elif self.reasoning_skill == "pronoun":
                logger.info("Generating Single Retrieval Benchmark with praphrased templates with pronoun")
                gen_func = self.single_retrieval_pronoun
            elif self.reasoning_skill == "rnidk":
                logger.info("Generating Single Retrieval Benchmark with standard baseline with randomly pruned idk")
                gen_func = self.single_retrieval_rnidk
            elif self.reasoning_skill == "storygen":
                raise KeyError("Story generation does not support single retrieval")
            else:
                raise NotImplementedError
        else:
            if self.reasoning_skill == 'standard':
                logger.info("Generating Multiple Retrieval Benchmark with no reasoning skills")
                gen_func = self.multiple_retrieval
            elif self.reasoning_skill == 'paraphrase':
                logger.info("Generating Multiple Retrieval Benchmark with praphrased templates")
                gen_func = self.multiple_retrieval_paraphrase
            elif self.reasoning_skill == "pronoun":
                logger.info("Generating Multiple Retrieval Benchmark with praphrased templates with pronoun")
                gen_func = self.multiple_retrieval_pronoun
            elif self.reasoning_skill == "rank":
                logger.info("Generating Single Retrieval Benchmark with mathematical ranking")
                gen_func = self.math_reasoning_rank
            elif self.reasoning_skill == "calculation":
                logger.info("Generating Single Retrieval Benchmark with mathematical calculation")
                gen_func = self.math_reasoning_calulation
            elif self.reasoning_skill == "twodiff":
                logger.info("Generating Single Retrieval Benchmark with mathematical twodiffning")
                gen_func = self.math_reasoning_twodiff
            elif self.reasoning_skill == "multihop":
                logger.info("Generating Single Retrieval Benchmark with multihop reasoning")
                gen_func = self.multihop_reasoning
            elif self.reasoning_skill == "storygen":
                logger.info("Generating Single Retrieval Benchmark with story generation")
                gen_func = self.multiple_retrieval_story_gen
            else:
                raise NotImplementedError
        for _ in tqdm(range(num_data)):
            self.dataset.append(gen_func())
            
            
    def to_json(self, output_folder):
        model_name = self.tokenizer_path.split('/')[-1]
        length = f"{self.named_length}k"
        if self.apply_fewshot:
            retri = f"{self.num_retrieval//3}_retrieval" # 2 shot
        else:
            retri = f"{self.num_retrieval}_retrieval"
        density = f"density_{str(self.density)}"
        position_start = int(self.position[0]*100)
        position_end = int(self.position[1]*100)
        position = f"position_{position_start}_{position_end}"
        output_dir = "_".join([model_name, length, retri, density, position])
        output_folder = os.path.join(output_folder, output_dir)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        file_name = f"data_{self.reasoning_skill}.json"
        output_file = os.path.join(output_folder, file_name)
        logger.info(f"Data saved to {output_file}")
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(self.dataset, f, ensure_ascii=False, indent=4)
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the BioQA dataset")
    # hyper-parameters that we used to control the dataset
    parser.add_argument('-l', '--length', type=int, default=8)
    parser.add_argument('-tok', '--tokenizer', type=str, default="/cpfs01/shared/XNLP_H800/hf_hub/Llama-3-8B-Instruct")
    parser.add_argument('-af', '--apply_fewshot', action='store_true', help="We perform 2-shot here")
    parser.add_argument('-nr', '--num_retrieval', type=int, default=1)
    parser.add_argument('-ps', '--position_start', type=float, default=0.0)
    parser.add_argument('-pe', '--position_end', type=float, default=1.0)
    parser.add_argument('-dd', '--distractor_density', type=float, default=1.0)
    parser.add_argument('-s', '--skill', type=str, choices=['standard', 'paraphrase', 'pronoun', "rnidk", "rank", "calculation", "twodiff", "multihop", "storygen"])
    parser.add_argument('-o', '--data_output_folder', type=str, required=True)
    parser.add_argument('-nd', '--num_data', type=int, default=800)
    args = parser.parse_args()
    logger.info(f"Args:{vars(args)}")


    benchmark = BioQABenchmarkGenerator(
        length=args.length,
        num_retrieval=args.num_retrieval,
        position=(args.position_start, args.position_end),
        density=args.distractor_density,
        num_bios=10000,
        tokenizer_path=args.tokenizer,
        reasoning_skill=args.skill,
        apply_fewshot=args.apply_fewshot
    )

    benchmark.generate(args.num_data)
    benchmark.to_json(args.data_output_folder)
