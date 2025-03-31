import argparse
from datetime import datetime
import os
from torch.utils.data import DataLoader
from loguru import logger
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from data_loader import LongContextDataSetFactory
from models import ModelsFactory
from utils import parse_config_from_data_path, save_json, load_json, split_list, custom_collate_fn
from evaluator import Evaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference experiments")
    parser.add_argument('-m', '--model_path', type=str, default="gpt-4o-mini-2024-07-18")
    parser.add_argument('-d', '--data_path', type=str, default='output/8k/context_questions_1001_1.json')
    parser.add_argument('-c', '--config_path', type=str, default='scripts/config/params.yaml')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help="The folder to store the output")
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-ae', '--apply_early_stop', action='store_true')
    parser.add_argument('-af', '--apply_fewshot', action='store_true')
    parser.add_argument('-db', '--debug', action='store_true')
    parser.add_argument('-rg', '--regenerate', action='store_true', help="if applied, the pipeline will ignore the previous results and regenerate a new one.")
    parser.add_argument('-eb', '--verbose_batch', type=int, default=2)

    parser.add_argument('-pn', '--port_name', type=int, default=None, help="port name for inference")
    parser.add_argument('-dp', '--data_parallel_num', type=int, default=1, help="number of data parallel, should be 1-8")

    
    args = parser.parse_args()
    if args.verbose_batch <= 1:
        args.verbose_batch = 1
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
    
    logger.info(f"Job started with args: {args}\ndata_configs:{data_configs}")

    llms = ModelsFactory(args)
    dataset = LongContextDataSetFactory(args, data_configs, model_path=args.model_path, data_path=args.data_path)

    # save the context dataset
    output_data_path = os.path.join(args.output_dir, 'data.json')
    save_json(output_data_path, dataset.full_data)
    logger.info(f"Resaved the data from {args.data_path} to {output_data_path}")

    dataset.prepare_dataset()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn)
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
        # Convert stored total_hits (stored as list) back to a numpy array for further accumulation.
        if inf_states.get("total_hits") is not None:
            inf_states["total_hits"] = np.array(inf_states["total_hits"])
    else:
        # Initialize states for a fresh run.
        inf_states = {
            "total_hits": None,
            "total_samples": 0,
            "acc": None,
            "last_batch_idx": -1,
            "batch_size": int(args.batch_size)
        }
        out = []
        start_batch = 0

    with ThreadPoolExecutor(max_workers=args.data_parallel_num) as executor:
        for batch_idx, (context_ids, q_ids, questions, prompts, labels) in enumerate(dataloader):
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

            # Define a helper function that will process a subbatch with the i-th llm.
            def process_subbatch(i):
                # Generate responses for the subbatch using llms[i]
                responses = llms[i].generate(split_prompts[i])
                # Evaluate and record the results for the subbatch.
                hit, results = evaluator.evaluate_and_record(
                    split_context_ids[i],
                    split_q_ids[i],
                    split_questions[i],
                    responses,
                    split_labels[i]
                )
                return hit, results, responses

            # Submit each non-empty subbatch to the executor.
            futures = {}
            for i in range(args.data_parallel_num):
                if len(split_prompts[i]) == 0:
                    continue
                futures[i] = executor.submit(process_subbatch, i)

            batch_hits = []
            batch_results = []
            # Retrieve the results in order (by llm index) to maintain the order in your global evaluation.
            for i in sorted(futures.keys()):
                hit, results, responses = futures[i].result()
                if args.debug:
                    logger.info(f"Subbatch {i} response:\n{responses}")
                batch_results.extend(results)
                # Ensure hit is a list for proper elementwise summation.
                if not isinstance(hit, list) and not isinstance(hit, tuple):
                    hit = [hit]
                batch_hits.append(hit)

            # Aggregate hit counts elementwise across subbatches.
            if batch_hits:
                hits_sum = np.sum(np.array(batch_hits), axis=0)
                if "total_hits" not in inf_states or inf_states["total_hits"] is None:
                    inf_states["total_hits"] = hits_sum
                else:
                    inf_states["total_hits"] += hits_sum

            inf_states["total_samples"] += len(batch_results)
            out.extend(batch_results)

            # Periodically log and save intermediate results.
            if batch_idx % args.verbose_batch == 0 or batch_idx == len(dataloader) - 1:
                saved_inf_states = {}
                cur_total_hit = [round(total_hit, 3) for total_hit in inf_states["total_hits"]]
                cur_acc = [round(acc, 3) for acc in (np.array(inf_states["total_hits"]) / inf_states["total_samples"]).tolist()]
                logger.info(f"Batch {batch_idx}: Acc {cur_acc}")

                saved_inf_states["total_hits"] = [int(h) for h in cur_total_hit]
                saved_inf_states['total_samples'] = inf_states["total_samples"]
                saved_inf_states["acc"] = [float(acc) for acc in cur_acc]
                saved_inf_states["last_batch_idx"] = int(batch_idx)
                saved_inf_states["batch_size"] = inf_states["batch_size"]
                saved_inf_states["args"] = vars(args)

                save_json(output_results_path, out)
                save_json(output_config_path, saved_inf_states)
                logger.info(f"Output and config update to {args.output_dir}")

            if args.debug:
                exit()

    out_sta = inf_states["acc"]
    logger.info(f"Generating finished. Acc: {out_sta}")
