import argparse
from datetime import datetime
import torch
torch.cuda.memory._set_allocator_settings('expandable_segments:False')
import json
import os

from torch.utils.data import DataLoader
from vllm import LLM
from vllm import SamplingParams
from loguru import logger

from data_loader import LongContextDataSetFactory
from models import ModelsFactory
from utils import parse_config_from_data_path, save_json
from evaluator import Evaluator
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference experiments")
    parser.add_argument('-m', '--model_path', type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument('-d', '--data_path', type=str, default='output/8k/context_questions_1001_1.json')
    parser.add_argument('-c', '--config_path', type=str, default='scripts/config/params.yaml')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="The folder to store the output")
    # parser.add_argument('-t', '--task', choices=['QA', 'TF'], default='QA')
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-ae', '--apply_early_stop', action='store_true')
    parser.add_argument('-af', '--apply_fewshot', action='store_true')
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('-eb', '--early_stop_batch', type=int, default=2)
    parser.add_argument('-es', '--early_stop_round', type=int, default=5, help="patience for early stop")
    
    args = parser.parse_args()
    if args.early_stop_batch <= 1:
        args.early_stop_batch = 1
    data_configs = parse_config_from_data_path(args.data_path)
    args.n_questions = data_configs['num_retrieval']
    args.max_token_length = data_configs['context_length']

    if args.apply_fewshot:
        args.output_dir += "_2shot"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_config_path = os.path.join(args.output_dir, 'config.json')
    output_results_path = os.path.join(args.output_dir, 'inference_results.json')
    log_file_name = datetime.now().strftime("%m-%d_%H-%M") + ".log"
    output_log = os.path.join(args.output_dir, log_file_name)
    logger.add(output_log)
    
    # logger.info(f"Job started with tasks: {args.task}")
    logger.info(f"Job started with args: {args}\ndata_configs:{data_configs}")

    llm = ModelsFactory(args)
    dataset = LongContextDataSetFactory(args, data_configs, model_path=args.model_path, data_path=args.data_path)

    # save the context dataset
    output_data_path = os.path.join(args.output_dir, 'data.json')
    save_json(output_data_path, dataset.full_data)
    logger.info(f"Resaved the data from {args.data_path} to {output_data_path}")

    out = []
    dataset.prepare_dataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    evaluator = Evaluator(args, dataset, data_configs)

    inf_states = {
        "total_hits": 0,
        "total_samples": 0,
        "acc": 0,
        "patience": 0,
        "args": vars(args)
    }

    for batch_idx, (context_ids, q_ids, question, prompts, labels) in enumerate(dataloader):
        logger.info(prompts)
        exit()
        if isinstance(labels[0], tuple):
            labels = [list(l) for l in zip(*labels)] #transpose the order of labels
        if args.debug:
            logger.info(prompts)
            logger.info(f"Labels:\n{labels}")
            responses = llm.generate(prompts)
            logger.info(f"Response:\n{responses}")
            hit, results = evaluator.evaluate_and_record(context_ids, q_ids, question, responses, labels)
            out.extend(results)
            logger.info(hit)
            exit()
        else:
            responses = llm.generate(prompts)
            hit, results = evaluator.evaluate_and_record(context_ids, q_ids, question, responses, labels)
            out.extend(results)

        inf_states["total_hits"] += hit
        inf_states["total_samples"] += len(results)
        if batch_idx % args.early_stop_batch == 0 or batch_idx==len(dataloader)-1:
            cur_acc = round(inf_states["total_hits"]/inf_states["total_samples"], 3)
            logger.info(f"Batch {batch_idx}: Acc {cur_acc}")
            if abs(cur_acc - inf_states["acc"])>=0.003: # allow 0.003 fluctuation
                inf_states["acc"] = cur_acc
                inf_states["patience"] = 1
            else:
                inf_states["patience"] += 1

            save_json(output_results_path, out)
            save_json(output_config_path, inf_states)
            logger.info(f"Output and config update to {args.output_dir}")

            if args.apply_early_stop and inf_states["patience"] >= args.early_stop_round:
                logger.info(f"Early stop at batch {batch_idx}")
                break
    out_sta = inf_states["acc"]
    logger.info(f"Generating finished. Acc: {out_sta}")
    