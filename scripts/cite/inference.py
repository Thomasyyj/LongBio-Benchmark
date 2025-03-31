import argparse
from datetime import datetime
import json
import os
import re
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger
import numpy as np

from concurrent.futures import ThreadPoolExecutor
from scripts.cite.cite_data_loader import LongContextCiteDataSetFactory
from scripts.models import ModelsFactory
from scripts.utils import parse_config_from_data_path, save_json, load_json, split_list
from scripts.evaluator import Evaluator

def custom_collate_fn(batch):
    context_ids, q_ids, question, prompts, labels, citations = [], [], [], [], [], []
    for data in batch:
        context_ids.append(data[0])
        q_ids.append(data[1])
        question.append(data[2])
        prompts.append(data[3])
        labels.append(data[4])
        citations.append(data[5])
    return context_ids, q_ids, question, prompts, labels , citations

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference experiments")
    parser.add_argument('-m', '--model_path', type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument('-d', '--data_path', type=str, default='output/8k/context_questions_1001_1.json')
    parser.add_argument('-c', '--config_path', type=str, default='scripts/config/params.yaml')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="The folder to store the output")
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-af', '--apply_fewshot', action='store_true')
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('-rg', '--regenerate', action='store_true', help="if applied, the pipeline will ignore the previous results and regenerate a new one.")
    parser.add_argument('-eb', '--verbose_batch', type=int, default=2)

    parser.add_argument('-pn', '--port_name', type=int, default=None, help="port name for inference")
    parser.add_argument('-dp', '--data_parallel_num', type=int, default=1, help="number of data parallel, should be 1-8")
    
    args = parser.parse_args()
    # for cite task we choose standard as backbone
    args.data_path = args.data_path.replace("cite", "standard")
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
    
    logger.info(f"Job started with args: {args}")

    llms = ModelsFactory(args)
    dataset = LongContextCiteDataSetFactory(
        args,
        model_path=args.model_path,
        data_path=args.data_path, 
        context_length=args.max_token_length
    )
    
    # save the context dataset
    output_data_path = os.path.join(args.output_dir, 'data.json')
    save_json(output_data_path, dataset.full_data)
    logger.info(f"Resaved the data from {args.data_path} to {output_data_path}")

    out = []
    dataset.prepare_dataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    data_configs = {'skill': 'cite'}
    evaluator = Evaluator(args, dataset, data_configs)

    # Check if a previous inference state exists and resume if so.
    if os.path.exists(output_config_path) and os.path.exists(output_results_path):
        inf_states = load_json(output_config_path)
        out = load_json(output_results_path)
        if inf_states['total_samples'] == 800 and not args.regenerate:
            logger.info("This task has already finished. Skip...")
            exit()
        if inf_states.get("batch_size"):
            previous_bs = inf_states["batch_size"]
            assert args.batch_size == previous_bs, f"Batch size should be {previous_bs}. {args.batch_size} given"
        else:
            inf_states["batch_size"] = int(args.batch_size)
        # last_batch_idx is stored as part of inf_states. We resume from the next batch.
        start_batch = inf_states.get("last_batch_idx", -1) + 1
        logger.info(f"Resuming inference from batch index {start_batch}.")
    else:
        # Initialize states for a fresh run.
        inf_states = {
            "total_ans_hits": 0,
            "total_cite_hits": 0,
            "total_samples": 0,
            "acc": 0,
            "cite_acc":0,
            "last_batch_idx": -1,
            "batch_size": int(args.batch_size),
            "args": vars(args)
        }
        out = []
        start_batch = 0

    
    with ThreadPoolExecutor(max_workers=args.data_parallel_num) as executor:
        for batch_idx, (context_ids, q_ids, questions, prompts, labels, citations) in enumerate(dataloader):
            if batch_idx < start_batch and not args.regenerate:
                continue
            if args.debug:
                logger.info(prompts)
                logger.info(f"Labels:\n{labels}")

            # Split each list into args.data_parallel_num parts.
            split_context_ids = split_list(context_ids, args.data_parallel_num)
            split_q_ids = split_list(q_ids, args.data_parallel_num)
            split_questions = split_list(questions, args.data_parallel_num)
            split_prompts = split_list(prompts, args.data_parallel_num)
            split_labels = split_list(labels, args.data_parallel_num)
            split_citations = split_list(citations, args.data_parallel_num)
            def process_subbatch(i):
                # Generate responses for the subbatch using llms[i]
                responses = llms[i].generate(split_prompts[i])
                # Evaluate and record the results for the subbatch.
                ans_hit, cite_hit, results = evaluator.evaluate_and_record(
                    split_context_ids[i],
                    split_q_ids[i],
                    split_questions[i],
                    responses,
                    split_labels[i],
                    split_citations[i]
                )
                return ans_hit, cite_hit, results, responses

            # Submit each non-empty subbatch to the executor.
            futures = {}
            for i in range(args.data_parallel_num):
                if len(split_prompts[i]) == 0:
                    continue
                futures[i] = executor.submit(process_subbatch, i)
            
            batch_ans_hits = []
            batch_cite_hits = []
            batch_results = []

            # Retrieve the results in order (by llm index) to maintain the order in your global evaluation.
            for i in sorted(futures.keys()):
                ans_hit, cite_hit, results, responses = futures[i].result()
                if args.debug:
                    logger.info(f"Subbatch {i} response:\n{responses}")
                batch_results.extend(results)
                batch_ans_hits.append(ans_hit)
                batch_cite_hits.append(cite_hit)

            ans_hits_sum = int(np.sum(np.array(batch_ans_hits), axis=0))
            cite_hits_sum = int(np.sum(np.array(batch_cite_hits), axis=0))

            inf_states["total_ans_hits"] += ans_hits_sum
            inf_states["total_cite_hits"] += cite_hits_sum
            inf_states["total_samples"] += len(batch_results)
            out.extend(batch_results)

            if batch_idx % args.verbose_batch == 0 or batch_idx==len(dataloader)-1:
                cur_acc = round(float(inf_states["total_ans_hits"]/inf_states["total_samples"]), 3)
                cur_cite_acc = round(float(inf_states["total_cite_hits"]/inf_states["total_samples"]), 3)
                logger.info(f"Batch {batch_idx}: Acc {cur_acc}, Cite Acc {cur_cite_acc}")
                inf_states["acc"] = cur_acc
                inf_states["cite_acc"] = cur_cite_acc
                inf_states["last_batch_idx"] = batch_idx

                save_json(output_results_path, out)
                save_json(output_config_path, inf_states)
                logger.info(f"Output and config update to {args.output_dir}")
            
            if args.debug:
                exit()

    out_sta = (inf_states["acc"], inf_states["cite_acc"])
    logger.info(f"Generating finished. Acc: {out_sta}")
    