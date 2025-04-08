import os
import json
import pandas as pd
import random

from typing import List
from __init__ import DATA_UTIL_PATH

attr_name_only_template = {
    'birthdate':'the birthday of <people>',
    'age': 'the age of <people>',
    'birthplace':'the birthplace of <people>', 
    'hobby':'the hobby of <people>', 
    'university':'the graduated university of <people>', 
    'major':'the major of <people>', 
    'work_city':'the city where <people> worked'
}

standard_questions_template = {
    'birthdate':'The birthday of <This person-o> is <birthdate>.',
    'age': 'The age of <This person-o> is <age>.',
    'birthplace':'<This person> was born in <birthplace>.', 
    'hobby':'The hobby of <This person-o> is <hobby>.', 
    'university':'The university where <This person> was graduated from is <university>.', 
    'major':'The major of <This person-o> was <major>.', 
    'work_city':'The city where <This person> worked is <work_city>.'
}

questions_template = {
    'birthdate':'What is the birthday of <This person-o>?',
    'age': 'What is the age of <This person-o>?',
    'birthplace':'What is the birthplace of <This person-o>?', 
    'hobby':'What is the hobby of <This person-o>?', 
    'university':'Which university did <This person> graduate from?', 
    'major':'What was the major of <This person-o>?', 
    'work_city':'Which city did <This person> work in?'
}

reasoning_question_template = {
    # 'rank' : "What is the order of {people_names} from youngest to oldest?",
    'rank': "Rank the following people in order of age from the youngest to the oldest: {people_names}", 
    'calculation': "What’s the age difference between {people_names}?",
    # 'twodiff': "Which people have an age difference of {age_diffence}? I only need one set of answers."
    'twodiff': "From the given profiles of these individuals, identify two individuals such that the age difference between them is {age_difference}. Only one pair of answers is needed."
}


def sample_boolen(density_prob, rng):
    return rng.random() < density_prob

def sample_choice(provided_list, out_sample_num, rng):
    return rng.choices(provided_list, k=out_sample_num)

def read_list_from_txt(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
         list = [line.strip() for line in f.readlines()]
    return list

class TemplateProcessor():
    def __init__(self, template_path) -> None:
        self.attribute_names = ['birthdate', 'birthplace', 'hobby', 'university', 'major', 'work_city']

        self.paraphrase_templates = {}
        for attribute in self.attribute_names:
            with open(f'{template_path}/{attribute}.json', 'r') as f:
                template_json = json.load(f)
            self.paraphrase_templates[attribute] = list(set(template_json['simple_paraphrases'] + template_json['complicated_paraphrases']))

        self.standard_templates = {attr:[template] for attr, template in standard_questions_template.items()}

    
    def get_template(self, name, rng=None, use_paraphrase=False) -> list:
        if use_paraphrase:
            templates = self.paraphrase_templates
        else:
            templates = self.standard_templates
        if rng is None:
            return random.choices(templates[name], k=1)
        return sample_choice(templates[name], 1, rng)  


class Bio():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.full_names = f"{self.first_name} {self.middle_name} {self.last_name}" if self.middle_name else f"{self.first_name} {self.last_name}"
        self.template_processor = TemplateProcessor(os.path.join(DATA_UTIL_PATH, "templates"))
        self.attribute_names = ['age'] + self.template_processor.attribute_names

        self.prefix = f'Below is the bio of {self.full_names}.'
        self.suffix = 'This is the end of this bio.'

        self.pron_prefix = f'Below is the self-introduction of {self.full_names}.'
        self.pron_suffix = 'This is the end of the introduction.'
        self.random_prune_rate = 0.2
              
    def __eq__(self, other):
        if isinstance(other, Bio):
            return self.first_name == other.first_name and self.last_name == other.last_name and self.middle_name == other.middle_name
        return False
    
    def __hash__(self):
        return hash((self.first_name, self.last_name, self.middle_name))
    
    @staticmethod
    def check_first_word(template, special_token):
        idx = template.find(special_token)
        if idx == 0 or template[:idx].strip()[-1] == '.':
            return True
        return False

    def fill_people_name(self, template, use_pronoun):
        if use_pronoun:
            proun_s = 'I'
            proun_o = "me"
            proun_p = "my"
        else:
            proun_p = "his/her"
        
        template = template.replace('their', proun_p).replace('Their', proun_p[0].upper()+proun_p[1:])

        if not use_pronoun:
            template = template.replace('<This person>', self.full_names).replace('<This person-o>', self.full_names)
            return template

        if '<This person>' in template:
            if self.check_first_word(template, '<This person>'):
                insert_proun = proun_s[0].upper() + proun_s[1:]
            else:
                insert_proun = proun_s
            template = template.replace('<This person>', insert_proun)
        elif '<This person-o>' in template:
            template = template.replace('<This person-o>', proun_o)
        else:
            assert False, "Should not go here"
        
        if self.check_first_word(template, "I's"):
            template = template.replace("I's", "My")
        else:
            template = template.replace("I's", "my")
        return template
    
    def to_description(
        self, 
        rng,
        exclude_attributions: List[str] = [],
        use_paraphrase: bool = False,
        use_pronoun: bool = False,
        random_attr: bool = False,
        freeze_attr: str = None,
        use_age: bool = False,
        ref_bio_name_and_attr: str = None
    ) -> dict:
        '''
        Ouput:
        {"descriptions":str, f"{attribute}": {"content":str, "position": [int, int]}}
        '''
        res = {}
        if use_pronoun:
            desc_prefix = self.pron_prefix
            desc_suffix = self.pron_suffix
        else:
            desc_prefix = self.prefix
            desc_suffix = self.suffix

        descriptions = desc_prefix + ' '
        for attribute in self.attribute_names:
            # age and birthdate should not appear together
            if use_age: 
                if attribute=='birthdate':
                    continue
            else:
                if attribute=='age':
                    continue
            if random_attr and sample_boolen(self.random_prune_rate, rng):
                if freeze_attr and freeze_attr==attribute:
                    pass
                else:
                    continue
            if attribute in exclude_attributions:
                continue
            start_idx = len(descriptions)
            if ref_bio_name_and_attr and ref_bio_name_and_attr[1] == attribute:
                attribute_ = attribute.replace("work_city", "work city")
                sampled_template = f'The {attribute_} of {self.full_names} is the same as {ref_bio_name_and_attr[0]}.'
            else:
                sampled_template = self.template_processor.get_template(attribute, rng, use_paraphrase)[0]
                sampled_template = self.fill_people_name(sampled_template, use_pronoun)
                sampled_template = sampled_template.replace(f'<{attribute}>', str(self.__getattribute__(attribute)))
            descriptions += sampled_template + ' '
            end_idx = len(descriptions)
            res[attribute] = {"content": self.__getattribute__(attribute), "position": (start_idx, end_idx)}
        descriptions += desc_suffix + ' '
        res['description'] = descriptions
        return res
    
    def to_question(self, attribute):  
        # this is just for retrival
        standard_template = standard_questions_template[attribute]
        standard_question = standard_template.replace("<This person-o>", self.full_names).replace("<This person>", self.full_names).replace(f"<{attribute}>", "")[:-1].strip()

        template = questions_template[attribute]
        question = template.replace("<This person-o>", self.full_names).replace("<This person>", self.full_names)

        return standard_question, question
    
    def to_sentance(self, attribute):  
        '''
        Given an attribute, output a sentance describing the attribute of this man
        e.g: 
        given: 'hobby'
        output: 'The hobby of Bob is basketball.'
        '''
        sampled_template = self.template_processor.get_template(attribute)[0]
        sampled_template = self.fill_people_name(sampled_template, use_pronoun=False)
        sampled_template = sampled_template.replace(f'<{attribute}>', str(self.__getattribute__(attribute)))
        return sampled_template
    
    def to_people_attr_only_description(self, attribute):
        '''
        Given an attribute, output a sentance with only hobby and people. Used for prompting story generation
        e.g: 
        given: 'hobby'
        output: 'The hobby of Bob.'
        '''
        return attr_name_only_template[attribute].replace('<people>', self.full_names)

