import random
import tiktoken

def sample_boolen(density_prob, rng):
    return rng.random() < density_prob

def sample_choice(provided_list, out_sample_num, rng):
    return rng.choices(provided_list, k=out_sample_num)

## gpt utils
def gpt_token_counter(messages, model="gpt-4o-mini-2024-07-18"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")

def join_names(names):
    """Used for join a list of names"""
    if not names:
        return ""

    # If there is only one name, just return it as is
    if len(names) == 1:
        return names[0]

    # If there are two names, join them with " and "
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"

    # If there are more than two names, join all but the last with ", "
    # and then append " and " followed by the last name.
    return f"{', '.join(names[:-1])} and {names[-1]}"


def generate_random_date(start_date, end_date):
    """
    start = "2023-01-01"
    end = "2023-12-31" 
    """
    import random
    from datetime import datetime, timedelta
    # convert to the string
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    days_between = (end_date - start_date).days
    rand_num = random.randint(0, days_between)
    return start_date + timedelta(days=rand_num)    

def read_list_from_txt(fpath):
    with open(fpath, 'r', encoding='utf-8') as f:
         name_list = [line.strip() for line in f.readlines()]
    return name_list

def compare_size(a, b):
    if a == b:
        return "="
    elif a < b:
        return "<"
    else:
        return ">"