import re
import json

def split_list(lst, n):
    k = len(lst)
    return [lst[i * k // n: (i+1) * k // n] for i in range(n)]

def custom_collate_fn(batch):
    context_ids, q_ids, question, prompts, labels = [], [], [], [], []
    for data in batch:
        context_ids.append(data[0])
        q_ids.append(data[1])
        question.append(data[2])
        prompts.append(data[3])
        labels.append(data[4])
    return context_ids, q_ids, question, prompts, labels 

def fill_template(template, **kwargs):
    """Fills in placeholders in a list of prompts.

    Args:
        prompts: A list of prompts, each a dictionary with 'role' and 'content' keys.
        given_context: The context to fill in.
        examples: The examples to fill in.
        question: The question to fill in.

    Returns:
      A list of filled prompts.
    """
    # Ensure all placeholders in the template have corresponding values in kwargs
    template_for_check = template if isinstance(template, str) else "\n".join([t["content"] for t in template])
    placeholders = re.findall(r'\{(\w+)\}', template_for_check)
    missing_placeholders = [ph for ph in placeholders if ph not in kwargs]
    if missing_placeholders:
        raise ValueError(f"Missing values for placeholders: {', '.join(missing_placeholders)}")
    
    if isinstance(template, str):
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"  # Create placeholder like {key}
            template = template.replace(placeholder, str(value))
        return template
    else:
        filled_prompts = []
        for key, value in kwargs.items():
            placeholder = f"{{{key}}}"
            for t in template:
                t["content"] = t["content"].replace(placeholder, str(value))
                filled_prompts.append(t)
        return filled_prompts

def parse_config_from_data_path(data_path):
    # e.g: longcontext/data/bios/glm-4-9b-chat-1m_8k_1_retrieval_density_1.0_position_0_100/data_paraphrase.json
    data_path = "_".join(data_path.split('/')[-2:])
    data_path = data_path.split('_')
    config = {}
    config['model'] = data_path[0]
    config['context_length'] = int(data_path[1][:-1])
    config['num_retrieval'] = int(data_path[2])
    config['density'] = float(data_path[5])
    config['position'] = [int(data_path[7]), int(data_path[8])]
    config['skill'] = data_path[-1].split('.json')[0]
    return config

def save_json(output_file, output):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    return

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)