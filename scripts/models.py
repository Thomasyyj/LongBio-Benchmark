import argparse
import time
import concurrent.futures
from omegaconf import OmegaConf
from openai import OpenAI

class ModelsFactory():
    def __new__(cls, args):
        if 'gpt' in args.model_path.lower():
            return [GPTModels(args)]
        else:
            if args.port_name is None:
                model_name = args.model_path.split('/')[-1]
                port = OmegaConf.load(args.config_path)["model_params"]['models_port'].get(model_name, 2025) #default to be 8888
            else:
                port = args.port_name
            return [OpenSourceModels(args,  port + n*100) for n in range(args.data_parallel_num)]

class OpenSourceModels():
    def __init__(self, args, port):
        self.model_path = args.model_path
        sampling_params_config= OmegaConf.load(args.config_path)['sampling_params']
        model_url = f"http://localhost:{port}/v1"
        self.client = OpenAI(
            base_url=model_url,
            api_key="token-abc123",
        )

        self.sampling_params = {
            "temperature":sampling_params_config['temperature'],
            "top_p":sampling_params_config['top_p'],
            "max_tokens":50*args.n_questions,
            "n":sampling_params_config['n_answer_sampled'],
            "seed":sampling_params_config['n_answer_sampled'],
        }

    def get_answer(self, client, model_name, prompt, sampling_params):
        max_duration = 180  # total retry duration in seconds (3 minutes)
        interval = 10       # wait time between retries in seconds
        start_time = time.time()
        
        while True:
            try:
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    **sampling_params
                )
                return completion.choices[0].text
            except ConnectionError as e:
                elapsed = time.time() - start_time
                if elapsed >= max_duration:
                    raise ConnectionError("Failed to connect after 3 minutes of retrying.") from e
                time.sleep(interval)

    def generate(self, prompts:list):
         num_parallel = len(prompts)
         results = [None for _ in range(num_parallel)]
         with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
             future_to_index = {}
             for index, question in enumerate(prompts):
                 future = executor.submit(
                     self.get_answer,
                     self.client,
                     self.model_path,
                     question,
                     self.sampling_params
                 )
                 future_to_index[future] = index
             for future in concurrent.futures.as_completed(future_to_index):
                 idx = future_to_index[future]
                 results[idx] = future.result()
         return results

class GPTModels(OpenSourceModels):
    def __init__(self, args):
        self.model_path = args.model_path.split('/')[-1]
        sampling_params_config= OmegaConf.load(args.config_path)['sampling_params']
        self.client = OpenAI()
        self.sleeptime = 30

        self.sampling_params = {
            "temperature":sampling_params_config['temperature'],
            "top_p":sampling_params_config['top_p'],
            "max_tokens":50*args.n_questions,
            "n":sampling_params_config['n_answer_sampled'],
            "seed":sampling_params_config['n_answer_sampled'],
        }

    def get_answer(self, client, model_name, messages, sampling_params):
        response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **sampling_params
            )
        time.sleep(1)
        return response.choices[0].message.content
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference experiments")
    parser.add_argument('-m', '--model_path', type=str, default="gpt-4o-mini-2024-07-18")
    args = parser.parse_args()
    models = OpenSourceModels(args)
    gpt_input = ["Tell me a joke."]
    output = models.generate(gpt_input, use_tqdm=True)
    print(output)
    