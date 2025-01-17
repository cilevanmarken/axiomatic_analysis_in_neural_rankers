# NOTE: set environment
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import glob
import argparse

from functools import partial
import transformer_lens.utils as utils
from transformer_lens import patching
from jaxtyping import Float
from itertools import product
from datetime import datetime

from helpers import (
    load_json_file,
    load_tokenizer_and_models,
    preprocess_queries,
    preprocess_corpus,
    encode_hf,
    encode_tl,
    compute_ranking_scores,
    create_df_from_nested_dict
)

from patching_helpers import (
    get_act_patch_block_every,
    get_act_patch_attn_head_out_all_pos,
    get_act_patch_attn_head_by_pos
)


def run_experiment(args):
    """
        - args.experiment_type: what type of experiment to run (options below)
            - block : patches the residual stream (before each layer), attention block, and MLP outputs across individual tokens
            - head_all : patches all attention heads individually across all tokens
            - head_pos : patches all attention heads individualls across individual tokens
            - labels : generates the tokenized documents
        - args.perturb_type : 
            - prepend : additional term is injected at the beginning of the document
            - append : additional term is injected at the end of the document
        - args.reduced_dataset : whether to use a reduced dataset
        - args.n_queries : number of queries to use if using a reduced dataset
        - args.n_docs : number of documents to use if using a reduced dataset
    """
    torch.set_grad_enabled(False)
    device = utils.get_device()
    print("Device:", device)

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # create folder structure to save results
    os.makedirs(f"{args.logging_folder}/results/{args.perturb_type}", exist_ok=True)
    os.makedirs(f"{args.logging_folder}/results_head_decomp/{args.perturb_type}", exist_ok=True)
    os.makedirs(f"{args.logging_folder}/results_attn_pattern/{args.perturb_type}", exist_ok=True)

    if args.dataset == "TFC2":
        os.makedirs(f"{args.logging_folder}/results/{args.perturb_type}/{args.TFC2_K}", exist_ok=True)
        os.makedirs(f"{args.logging_folder}/results_head_decomp/{args.perturb_type}/{args.TFC2_K}", exist_ok=True)
        os.makedirs(f"{args.logging_folder}/results_attn_pattern/{args.perturb_type}/{args.TFC2_K}", exist_ok=True)

    # Load files
    if args.dataset == "TFC1-I":
        baseline_path = f"data/{args.dataset}/TFC1-I_{args.perturb_type}_baseline.json"
        perturbed_path = f"data/{args.dataset}/TFC1-I_{args.perturb_type}_corpus.json"
    elif args.dataset == "TFC1-R":
        # baseline is expected to rank higher, so we use the reverse
        baseline_path = f"data/{args.dataset}/TFC1-R_corpus.json"
        perturbed_path = f"data/{args.dataset}/TFC1-R_baseline.json"
    elif args.dataset == "TFC2":
        baseline_path = f"data/{args.dataset}/TFC2_{args.TFC2_K}_baseline.json"
        perturbed_path = f"data/{args.dataset}/TFC2_{args.TFC2_K}_corpus.json"
    
    # tfc1_precomputed_scores = pd.read_csv(os.path.join(fbase_path, "tfc1_add_{}_target_qids_scores.csv".format(args.perturb_type)))
    tfc1_add_queries = pd.read_csv(os.path.join("data", "QIDs_with_text.csv"), header=None, names=["_id", "text"])
    tfc1_add_baseline_corpus = load_json_file(baseline_path)["corpus"]
    tfc1_add_dd_corpus = load_json_file(perturbed_path)["corpus"]

    target_qids = tfc1_add_queries["_id"].tolist()
    tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)]
    if args.reduced_dataset:
        target_qids = random.sample(tfc1_add_queries["_id"].tolist(), args.n_queries)
        tfc1_add_queries = tfc1_add_queries[tfc1_add_queries["_id"].isin(target_qids)]

    pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer, _, tl_model = load_tokenizer_and_models(pre_trained_model_name, device)

    # create pd dataframe from nested dict
    if args.skip_already_computed:
        if args.dataset == "TFC2":
            computed_results = pd.read_csv(f"data/{args.dataset}/computed_results_{args.perturb_type}_{args.TFC2_K}.csv")
        else:
            computed_results = pd.read_csv(f"data/{args.dataset}/computed_results_{args.perturb_type}.csv")
    else:
        computed_results = create_df_from_nested_dict(tfc1_add_dd_corpus)

    # Preprocess queries
    queries_dataloader = preprocess_queries(tfc1_add_queries, tokenizer)

    # Loop through each query and run activation patching
    for i, qid in enumerate(tqdm(target_qids)):

        # Get query embedding
        q_tokenized = list(filter(lambda item: item["_id"] == qid, queries_dataloader))[0]
        q_outputs = tl_model(
            q_tokenized["input_ids"],
            return_type="embeddings",
            one_zero_attention_mask=q_tokenized["attention_mask"],
        )
        q_embedding = q_outputs[:,0,:].squeeze(0) #.detach().cpu().numpy() # leave on device
        
        # Get and preprocess documents
        target_docs = tfc1_add_dd_corpus[str(qid)]
        if args.reduced_dataset:
            # get first n docs
            target_doc_ids = list(target_docs.keys())[:args.n_docs]
            target_docs = {doc_id: target_docs[doc_id] for doc_id in target_doc_ids}
        corpus_dataloader = preprocess_corpus(target_docs, tokenizer)

        # batch size = 1 doc
        for j, batch in enumerate(corpus_dataloader):
            print(f"QUERY @ {i}, DOC @ {j}")
            try:

                # skip if already computed
                if args.skip_already_computed:
                    mask = (computed_results['qid'] == str(qid)) & (computed_results['doc_id'] == str(batch["_id"][0]))
                    if not computed_results[mask].empty:
                        print(f"Skipping query {qid} and doc {batch['_id'][0]}")
                        continue

                # Get baseline doc
                doc_id = batch["_id"][0]
                baseline_doc = tfc1_add_baseline_corpus[str(qid)][doc_id]["text"]
                baseline_tokens = tokenizer(baseline_doc, truncation=True, return_tensors="pt")

                # Run perturbed prompt with cache to store activations
                perturbed_embeddings, perturbed_cache = tl_model.run_with_cache(
                    batch["input_ids"],
                    one_zero_attention_mask=batch["attention_mask"],
                    return_type="embeddings",
                )
                perturbed_embedding = perturbed_embeddings[:,0,:].squeeze(0) #.detach().cpu().numpy()

                # Check lengths of pertubred and baseline tokens and adjust if needed
                p_len = torch.sum(batch["attention_mask"])
                b_len = torch.sum(baseline_tokens["attention_mask"])

                adj_n = p_len - b_len
                cls_tok = baseline_tokens["input_ids"][0][0]
                sep_tok = baseline_tokens["input_ids"][0][-1]

                # Hacky thing b/c of the way the diagnostic dataset was created (it was originally created just for prepend)
                # So will always need to adjust the tokens for append
                if args.perturb_type == "append":
                    filler_tokens = torch.full((adj_n+1,), baseline_tokens["input_ids"][0][1]) # skip CLS token
                    filler_attn_mask = torch.full((adj_n+1,), baseline_tokens["attention_mask"][0][1]) 
                    adj_doc = torch.cat((baseline_tokens["input_ids"][0][2:-1], filler_tokens))
                    baseline_tokens["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
                    baseline_tokens["attention_mask"] = torch.cat((baseline_tokens["attention_mask"][0][1:], filler_attn_mask), dim=0).view(1,-1)
                elif args.perturb_type == "prepend":
                    # But for prepend, we only need to adjust if the lengths are different
                    if p_len != b_len:
                        filler_tokens = torch.full((adj_n,), baseline_tokens["input_ids"][0][1]) # skip CLS token
                        filler_attn_mask = torch.full((adj_n,), baseline_tokens["attention_mask"][0][1]) 
                        adj_doc = torch.cat((filler_tokens, baseline_tokens["input_ids"][0][1:-1]))
                        baseline_tokens["input_ids"] = torch.cat((cls_tok.view(1), adj_doc, sep_tok.view(1)), dim=0).view(1,-1)
                        baseline_tokens["attention_mask"] = torch.cat((filler_attn_mask, baseline_tokens["attention_mask"][0]), dim=0).view(1,-1)
                

                # Get baseline doc embedding
                baseline_outputs = tl_model(
                    baseline_tokens["input_ids"],
                    return_type="embeddings",
                    one_zero_attention_mask=baseline_tokens["attention_mask"],
                )
                baseline_embedding = baseline_outputs[:,0,:].squeeze(0) #.detach().cpu().numpy()

                # Get scores for baseline and perturbed documents
                baseline_score = torch.matmul(q_embedding, baseline_embedding.t())
                # NOTE: this is precomputed, 
                # perturbed_score = torch.tensor(tfc1_precomputed_scores[(tfc1_precomputed_scores["qid"] == qid) & (tfc1_precomputed_scores["doc_id"] == int(doc_id))]["p_score"].item())
                perturbed_score = torch.matmul(q_embedding, perturbed_embedding.t())

                # save these scores
                if args.save_intermediate:
                    mask = (computed_results['qid'] == str(qid)) & (computed_results['doc_id'] == str(doc_id))
                    computed_results.loc[mask, "og_score"] = baseline_score.item()
                    computed_results.loc[mask, "p_score"] = perturbed_score.item()
                    computed_results.loc[mask, "score_diff"] = perturbed_score.item() - baseline_score.item()
                    computed_results.loc[mask, "percent_change"] = (perturbed_score.item() - baseline_score.item()) / baseline_score.item()

                '''
                Linear function of score diff, calibrated so that it equals 0 when performance is 
                same as on clean input, and 1 when performance is same as on corrupted input.
                '''
                def ranking_metric(patched_doc_embedding, og_score=baseline_score, p_score=perturbed_score):
                    patched_score = torch.matmul(q_embedding, patched_doc_embedding.t())
                    return (patched_score - og_score) / (p_score - og_score)
                
                # Patch after each layer (residual stream, attention, MLPs)
                if args.experiment_type == "block":
                    if args.dataset == "TFC2":
                        result_file = f"{args.logging_folder}/results/{args.perturb_type}/{args.TFC2_K}/{qid}_{doc_id}_block.npy"
                    else:
                        result_file = f"{args.logging_folder}/results/{args.perturb_type}/{qid}_{doc_id}_block.npy"

                    # skip if already computed
                    if args.skip_already_computed and os.path.exists(result_file):
                        print(f"Skipping {qid}_{doc_id}")
                        continue

                    act_patch_block_every = get_act_patch_block_every(
                        tl_model,
                        device,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric
                    )
                    detached_block_results = act_patch_block_every.detach().cpu().numpy()
                    if args.save: np.save(result_file, detached_block_results)
                    

                # Patch attention heads
                elif args.experiment_type == "head_all":
                    if args.dataset == "TFC2":
                        result_file = f"{args.logging_folder}/results_head_decomp/{args.perturb_type}/{args.TFC2_K}/{qid}_{doc_id}_head.npy"
                    else:
                        result_file = f"{args.logging_folder}/results_head_decomp/{args.perturb_type}/{qid}_{doc_id}_head.npy"

                    # skip if already computed
                    if args.skip_already_computed and os.path.exists(result_file):
                        print(f"Skipping {qid}_{doc_id}")
                        continue

                    act_patch_attn_head_out_all_pos = get_act_patch_attn_head_out_all_pos(
                        tl_model,
                        device,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric
                    )
                    detached_head_results = act_patch_attn_head_out_all_pos.detach().cpu().numpy()
                    if args.save: np.save(result_file, detached_head_results)


                # Patch heads by position
                elif args.experiment_type == "head_pos":
                    if args.dataset == "TFC2":
                        raise NotImplementedError("TFC2 not implemented for head_attn, do not know which heads are important")
                        result_file = f"{args.logging_folder}/results_head_decomp/{args.perturb_type}/{args.TFC2_K}/{qid}_{doc_id}_head_by_pos.npy"
                    else:
                        result_file = f"{args.logging_folder}/results_head_decomp/{args.perturb_type}/{qid}_{doc_id}_head_by_pos.npy"

                    # skip if already computed
                    if args.skip_already_computed and os.path.exists(result_file):
                        print(f"Skipping {qid}_{doc_id}")
                        continue

                    layer_head_list = [(0,9), (1,6), (2,3), (3,8)]
                    act_patch_attn_head_out_by_pos = get_act_patch_attn_head_by_pos(
                        tl_model,
                        device,
                        baseline_tokens,
                        perturbed_cache,
                        ranking_metric,
                        layer_head_list
                    )
                    detached_head_pos_results = act_patch_attn_head_out_by_pos.detach().cpu().numpy()
                    if args.save: np.save(result_file, detached_head_pos_results)


                # Get attention patterns for head
                elif args.experiment_type == "head_attn":
                    attn_heads = [(0,9), (1,6), (2,3), (3,8)]
                    for layer, head in attn_heads:
                        attn_pattern = perturbed_cache["pattern", layer][:,head].mean(0).detach().cpu().numpy()

                        if args.dataset == "TFC2":
                            raise NotImplementedError("TFC2 not implemented for head_attn, do not know which heads are important")
                            if args.save: np.save(f"{args.logging_folder}/results_attn_pattern/{args.perturb_type}/{args.TFC2_K}/{qid}_{doc_id}_{layer}_{head}_attn_pattern.npy", attn_pattern)
                        else:
                            if args.save: np.save(f"{args.logging_folder}/results_attn_pattern/{args.perturb_type}/{qid}_{doc_id}_{layer}_{head}_attn_pattern.npy", attn_pattern)

                
                elif args.experiment_type == "labels":
                    decoded_tokens = [tokenizer.decode(tok) for tok in batch["input_ids"][0]]
                    labels = ["{} {}".format(tok,i) for i, tok in enumerate(decoded_tokens)]

                    if args.save: 
                        if args.dataset == "TFC2":
                            with open(f"{args.logging_folder}/results/{args.perturb_type}/{args.TFC2_K}/{qid}_{doc_id}_labels.txt", "w", encoding='utf-8') as f:
                                for item in labels:
                                    f.write(str(item) + '\n')
                        else:
                            with open(f"{args.logging_folder}/results/{args.perturb_type}/{qid}_{doc_id}_labels.txt", "w", encoding='utf-8') as f:
                                for item in labels:
                                    f.write(str(item) + '\n')


            except Exception as e:
                print(f"ERROR: {e} for query {qid} and document {doc_id}")

            # save intermediate results
            if args.save_intermediate:
                computed_results.to_csv(f"temp_intermediate_results.csv", index=False)

    # compute results and save
    computed_results['og_rank'] = computed_results.groupby('qid')['og_score'].rank(
        method='min',  # Use minimum rank for ties
        ascending=False,  # Higher scores get lower ranks (rank 1 is best)
        na_option='bottom'  # Put NaN values at the bottom
    )
    computed_results['p_rank'] = computed_results.groupby('qid')['p_score'].rank(
        method='min',  # Use minimum rank for ties
        ascending=False,  # Higher scores get lower ranks (rank 1 is best)
        na_option='bottom'  # Put NaN values at the bottom
    )
    computed_results['change_in_rank'] = computed_results['og_rank'] - computed_results['p_rank']

    # save
    if args.save_intermediate:
        if args.dataset == "TFC2":
            computed_results.to_csv(f"data/{args.dataset}/computed_results_{args.perturb_type}_{args.TFC2_K}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        else:
            computed_results.to_csv(f"data/{args.dataset}/computed_results_{args.perturb_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run activation patching with the specific patching experiment and perturbation types.")
    parser.add_argument("--dataset", default="TFC2", choices=["TFC1-I", "TFC1-R", "TFC2"])
    parser.add_argument("--experiment_type", default="block", choices=["block", "head_all", "head_pos", "head_attn", "labels"], 
                        help="What will be patched (e.g., block).")
    parser.add_argument("--perturb_type", default="append", choices=["append", "prepend"], 
                        help="The perturbation to apply (e.g., append).")
    parser.add_argument("--TFC2_K", default=5, type=int, choices=[1, 2, 5, 10, 50], help="K")
    parser.add_argument("--save_intermediate", default=True, help="Save intermediate results.")
    
    # reduced dataset
    parser.add_argument("--reduced_dataset", default=False, action='store_true',
                        help="Whether to use a reduced dataset.")
    parser.add_argument("--n_queries", type=int, default=1, 
                        help="Number of queries to use if using a reduced dataset.")
    parser.add_argument("--n_docs", type=int, default=3, 
                        help="Number of documents to use if using a reduced dataset.")
    
    # reproducibility
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--skip_already_computed", default=True, action='store_true', help="Do not overwrite already computed files and do not recompute results.")
    parser.add_argument("--save", default=True, help="Save results.")

    args = parser.parse_args()

    args.logging_folder = f"./results/{args.dataset}"
    
    _ = run_experiment(args)
