import re
from torch import Tensor
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer

from scripts.models import OpenSourceModels

class Evaluator:
    def __new__(cls, args, datasets, data_configs, **kwargs):
        use_few_shot = args.apply_fewshot
        if data_configs['skill'] == 'twodiff':
            return AgeDiffEvaluator(datasets, use_few_shot)
        elif data_configs['skill'] == 'IDK':
            return IDKEvaluator(datasets, use_few_shot=False, **kwargs)
        elif data_configs['skill'] == 'cite':
            return CitationEvaluator(datasets, use_few_shot)
        else:
            return GeneralEvaluator(datasets, use_few_shot)

class GeneralEvaluator():
    def __init__(self, datasets, use_few_shot):
        if "instruct" not in datasets.model_path.lower():
            self.use_base_model = True
        else:
            self.use_base_model = False
        self.datasets = datasets
        self.use_few_shot = use_few_shot

    def evaluate_and_record(self, context_indices, q_indices, question, preds, refs):
        hit = 0
        results = []
        for i in range(len(preds)):
            answer = preds[i]
            if self.use_base_model:
                answer = answer[:answer.find("Question: ")] # truncate the answer
            ref = refs[i]
            if isinstance(ref, Tensor):
                ref = int(ref)
            hit_ = self.cal_hit(answer, ref)
            hit += hit_
            results.append({"datapoint": {
                "context_idx": int(context_indices[i]), 
                "q_idx": int(q_indices[i]),
                "question": question[i]
            }, "ref": ref, "pred": answer, "hit": hit_}) 
        return hit, results

    def cal_hit(
        self,
        pred, 
        ref
    ):
        if isinstance(ref, str): 
            if ref.strip().lower() not in pred.lower():
                return 0
            return 1
        elif isinstance(ref, int):
            pred = self.extract_number(pred)
            if pred and int(ref) == pred:
                return 1
            else:
                return 0
        else:
            # logger.info(f"Ref is not a string, but a list: {ref}. So we are in the multi-retrieval mode.")
            current_pos = 0
            for r in ref:
                pattern = re.escape(r)
                match = re.search(pattern, pred[current_pos:], re.IGNORECASE)
                if not match:
                    return 0
                # Update current_pos to the end of the current match relative to the full string.
                current_pos += match.end()
            return 1
    
    def extract_number(self, answer_string):
        if not isinstance(answer_string, str):
            return answer_string
        matches = re.findall(r'\d+', answer_string)
        if self.use_few_shot:
            return int(matches[-1]) if matches else None
        else:
            return int(matches[-1]) if matches else None # 
    

class AgeDiffEvaluator(GeneralEvaluator):
    def evaluate_and_record(self, context_indices, q_indices, question, preds, refs):
        hit = 0
        results = []
        for i in tqdm(range(len(preds))):
            context = self.datasets.full_data[context_indices[i]]['context']
            answer = preds[i]
            ref = int(refs[i]) # transform the tensor into integer
            age_diffs = self.get_age_diffs(answer, context)
            logger.debug(f"Pred: {age_diffs}")
            logger.debug(f"ref: {ref}")
            if age_diffs:
                hit_ = self.cal_hit(age_diffs, ref)
            else:
                hit_ = 0
            hit += hit_
            results.append({"datapoint": {
                "context_idx": int(context_indices[i]), 
                "q_idx": int(q_indices[i]),
                "question": question[i]
            }, "ref": ref, "pred": answer, "pred_age_diff": age_diffs, "hit": hit_}) 
        return hit, results
    
    def get_age_diffs(self, answer, context_list):
        # extract the name in the answer
        name_pattern = r"(?<=Below is the bio of )(.*?)(?=\.)"
        ages = []
        idendified_names = []
        names = []
        for context in context_list:
            name_match = re.findall(name_pattern, context)
            if name_match and name_match[0] in answer:
                name = name_match[0]
                if name in idendified_names:
                    continue
                idendified_names.append(name)
                age_pattern = rf"The age of {name} is (\d+)."
                age_match = re.search(age_pattern, context)
                ages.append(int(age_match.group(1)))
        if len(ages) != 2:
            return None
        return int(abs(ages[0] - ages[1])) 

class IDKEvaluator(GeneralEvaluator):
    def __init__(self, datasets, use_few_shot, ori_results):
        super().__init__(datasets, use_few_shot=False) 
        self.ori_results = ori_results
    
    def evaluate_and_record(self, context_indices, q_indices, question, preds, refs):
        hit = 0
        ori_hit = 0
        idk_hit = 0
        results = []
        for i in range(len(preds)):
            answer = preds[i]
            if self.use_base_model:
                answer = answer[:answer.find("Question: ")] # truncate the answer
            ref = refs[i]
            if isinstance(ref, Tensor):
                ref = int(ref)
            idk_hit_ = self.cal_hit(answer, ref)
            ori_hit_ = self.ori_results[context_indices[i]].get('hit')
            if ori_hit_ is None:
                ori_hit_ = 1 if self.ori_results[context_indices[i]]['ref'] in self.ori_results[context_indices[i]]['pred'] else 0
            hit_ = idk_hit_ and ori_hit_
            ori_hit += ori_hit_
            idk_hit += idk_hit_
            hit += hit_
            results.append({"datapoint": {
                "context_idx": int(context_indices[i]), 
                "q_idx": int(q_indices[i]),
                "question": question[i]
            }, "ref": ref, "pred": answer, "idk_hit": idk_hit_, 'ori_hit': ori_hit_, 'hit': hit_}) 
        return (ori_hit, idk_hit, hit), results
    
class CitationEvaluator(GeneralEvaluator):
    def evaluate_and_record(self, context_indices, q_indices, question, preds, refs, citations):
        ans_hit = 0
        cite_hit = 0
        results = []
        for i in range(len(preds)):
            answer = preds[i]
            ref = refs[i]
            citation = citations[i]
            answer_hit = self.cal_hit(answer, ref)
            citation_hit = self.cal_hit(answer, citation)
            ans_hit += answer_hit
            cite_hit += citation_hit
            results.append({"datapoint": {
                "context_idx": int(context_indices[i]), 
                "q_idx": int(q_indices[i]),
                "question": question[i]
            }, "answer_ref": ref, "citation_ref":citation, "pred":answer, "answer_hit":answer_hit, "citation_hit":citation_hit}) 
        return ans_hit, cite_hit, results

    def cal_hit(
        self,
        pred, 
        ref
    ):
        if isinstance(ref, str): 
            if ref.strip().lower() not in pred.lower():
                return 0
            return 1
        elif len(ref) == 1:
            if ref[0].strip().lower() not in pred.lower():
                return 0
            return 1
        else:
            # logger.info(f"Ref is not a string, but a list: {ref}. So we are in the multi-retrieval mode.")
            current_pos = 0
            for r in ref:
                pattern = re.escape(r)
                match = re.search(pattern, pred[current_pos:], re.IGNORECASE)
                if not match:
                    return 0
                # Update current_pos to the end of the current match relative to the full string.
                current_pos += match.end()
            return 1