# NOTE: set environment
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.subplots as sp
import os
import glob
import json
import argparse
from transformers import AutoTokenizer


def load_jsonl_file_into_dict(fname):
    """Load a JSONL file into a Python dictionary."""
    data = {}
    with open(fname, "r") as f:
        for line in f:
            loaded_line = json.loads(line)
            iden = loaded_line["_id"]
            text = loaded_line["text"]
            data[iden] = text

    return data


def load_label_file(file_path):
    """Load a label file."""
    # NOTE: changed encoding from utf-8 to ascii
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    words = []
    for line in lines:
        word, _ = line.strip().split()
        words.append(word)

    return words


def load_doc_results_by_rank(
    scores_fname,
    result_type,
    results_path,
    labels_path,
    args,
    rank_start=1,
    rank_end=10,
    rank_type="og_rank",
):
    """Load results for a specific rank range."""
    # Get query, doc ids. If TFC2: filter out p_score < og_score
    if args.dataset == "TFC2" and args.filter:
        scores = pd.read_csv(
            scores_fname, usecols=["qid", "doc_id", "score_diff", rank_type]
        )
        scores = scores[scores["score_diff"] > 0]
    else:
        scores = pd.read_csv(scores_fname, usecols=["qid", "doc_id", rank_type])
    filtered_scores = scores[
        scores[rank_type].isin(list(range(rank_start, rank_end + 1)))
    ]
    target_docs = list(filtered_scores[["qid", "doc_id"]].to_records(index=False))

    # Load files
    results, labels, qids = [], [], []
    for qid, doc_id in target_docs:
        result_fname = os.path.join(
            results_path, "{}_{}_{}.npy".format(qid, doc_id, result_type)
        )

        if os.path.exists(result_fname):
            result_data = np.load(result_fname)
            results.append(result_data)

            if result_type == "block" or result_type == "head_by_pos":
                label_fname = os.path.join(
                    labels_path, "{}_{}_labels.txt".format(qid, doc_id)
                )
                label_data = load_label_file(label_fname)
                labels.append(label_data)

            qids.append(str(qid))

    return results, labels, qids


def segment_tokens(
    label, perturb_type, qid, doc_id, full_q_dict, selected_terms_dict, tokenizer
):
    """Segment docs into token classes."""
    # Get query data
    qid_lookup = {}
    if qid not in qid_lookup:
        full_q, selected_term = get_query_data(qid, full_q_dict, selected_terms_dict)
        qid_lookup[qid] = {
            "full_query_toks": [tokenizer.tokenize(term) for term in full_q.split()],
            "selected_term_toks": tokenizer.tokenize(selected_term),
        }

    selected_term_toks = qid_lookup[qid]["selected_term_toks"]
    full_q_tok_list = qid_lookup[qid]["full_query_toks"]

    # CLS token idx
    cls_idx = 0

    # Get inj term idxs
    if perturb_type == "prepend":
        og_doc_start_idx = len(selected_term_toks) + 1
        inj_range = range(1, og_doc_start_idx)
        og_doc_end_idx = len(label) - 1
    elif perturb_type == "append":
        og_doc_start_idx = 1
        og_doc_end_idx = len(label) - len(selected_term_toks) - 1  # not inclusive
        inj_range = range(og_doc_end_idx, len(label) - 1)
    inj_idxs = [*inj_range]

    # first_q_term_inj_idxs = []
    q_term_inj_idxs = []
    q_term_non_inj_idxs = []
    for query_tokens in full_q_tok_list:
        # tok_idxs = []
        inj_tok_idxs = []
        non_inj_tok_idxs = []

        # strict query term matching
        for i in range(og_doc_start_idx, og_doc_end_idx + 1 - len(query_tokens)):
            window = label[i : i + len(query_tokens)]
            if window == query_tokens:
                window_range = range(i, i + len(query_tokens))

                if window == selected_term_toks:
                    # if not first_q_term_inj_idxs:
                    #     first_q_term_inj_idxs = inj_tok_idxs + [*window_range]
                    # else:
                    inj_tok_idxs = inj_tok_idxs + [*window_range]
                else:
                    non_inj_tok_idxs = non_inj_tok_idxs + [*window_range]

        q_term_inj_idxs = q_term_inj_idxs + inj_tok_idxs
        q_term_non_inj_idxs = q_term_non_inj_idxs + non_inj_tok_idxs

    # Get non query terms
    all_doc_idxs = list(range(og_doc_start_idx, og_doc_end_idx))
    non_q_term_idxs = list(
        set(all_doc_idxs) - set(q_term_inj_idxs) - set(q_term_non_inj_idxs)
    )  # - set(first_q_term_inj_idxs))

    # SEP token
    sep_idx = len(label) - 1

    return (
        cls_idx,
        inj_idxs,
        q_term_inj_idxs,
        q_term_non_inj_idxs,
        non_q_term_idxs,
        sep_idx,
    )  # , first_q_term_inj_idxs


def load_head_pattern_by_rank(
    results_path,
    labels_path,
    perturb_type,
    score_df,
    heads,
    rank_start,
    rank_end,
    full_q_dict,
    selected_terms_dict,
    tokenizer,
):
    """Load head attention patterns for a specific rank range."""
    # Get query, doc ids
    filtered_scores = score_df[
        score_df["og_rank"].isin(list(range(rank_start, rank_end + 1)))
    ]
    target_docs = list(filtered_scores[["qid", "doc_id"]].to_records(index=False))

    results = {}
    exists_results = {}
    not_exists_results = {}
    for qid, doc_id in target_docs:
        # Get label and segment
        label_fname = os.path.join(labels_path, "{}_{}_labels.txt".format(qid, doc_id))
        label = load_label_file(label_fname)

        (
            cls_idx,
            inj_idxs,
            q_term_inj_idxs,
            q_term_non_inj_idxs,
            non_q_term_idxs,
            sep_idx  # ,
            # first_q_term_inj_idxs
        ) = segment_tokens(
            label,
            perturb_type,
            str(qid),
            str(doc_id),
            full_q_dict,
            selected_terms_dict,
            tokenizer,
        )

        # Get results for each head and segment
        for head in heads:
            fname = os.path.join(
                results_path,
                "{}_{}_{}_{}_attn_pattern.npy".format(qid, doc_id, head[0], head[1]),
            )
            data = np.load(fname)

            softmax_attn = data  # attention pattern is already softmaxed

            all_other_tok_idxs = list(
                set(range(len(label))) - set(inj_idxs) - set([sep_idx])
            )

            inj_to_cls_attn = np.mean(softmax_attn[inj_idxs][:, cls_idx])
            if not q_term_inj_idxs:
                inj_to_qterm_inj_attn = 0.0
                qterm_inj_to_inj = 0.0
            else:
                inj_to_qterm_inj_attn = np.mean(
                    softmax_attn[inj_idxs][:, q_term_inj_idxs]
                )
                qterm_inj_to_inj = np.mean(softmax_attn[q_term_inj_idxs][:, inj_idxs])
            if not q_term_non_inj_idxs:
                inj_to_qterm_non_inj_attn = 0.0
            else:
                inj_to_qterm_non_inj_attn = np.mean(
                    softmax_attn[inj_idxs][:, q_term_non_inj_idxs]
                )
            inj_to_other_attn = np.mean(softmax_attn[inj_idxs][:, non_q_term_idxs])
            inj_to_sep_attn = np.mean(softmax_attn[inj_idxs][:, sep_idx])
            all_other_to_sep_attn = np.mean(
                softmax_attn[all_other_tok_idxs][:, sep_idx]
            )

            # segment data - should be single values
            cls_tok_avg = np.mean(softmax_attn[:, cls_idx])
            inj_toks_avg = np.mean(softmax_attn[:, inj_idxs].flatten())
            if not q_term_inj_idxs:
                q_term_inj_toks_avg = 0
            else:
                q_term_inj_toks_avg = np.mean(
                    softmax_attn[:, q_term_inj_idxs].flatten()
                )
            if not q_term_non_inj_idxs:
                q_term_non_inj_toks_avg = 0
            else:
                q_term_non_inj_toks_avg = np.mean(
                    softmax_attn[:, q_term_non_inj_idxs].flatten()
                )
            non_q_term_toks_avg = np.mean(softmax_attn[:, non_q_term_idxs].flatten())
            sep_tok_avg = np.mean(softmax_attn[:, sep_idx])

            result = [
                inj_to_qterm_inj_attn,
                qterm_inj_to_inj,
                inj_to_qterm_non_inj_attn,
                inj_to_sep_attn,
                all_other_to_sep_attn,
            ]

            # add to results
            if head not in results:
                results[head] = [result]
                if not q_term_inj_idxs:
                    exists_results[head] = []
                    not_exists_results[head] = [result]
                else:
                    exists_results[head] = [result]
                    not_exists_results[head] = []
            else:
                results[head].append(result)
                if not q_term_inj_idxs:
                    not_exists_results[head].append(result)
                else:
                    exists_results[head].append(result)

    avg_results = []
    exists_avg_results = []
    not_exists_avg_results = []
    for head in results:
        avg_result = np.mean(np.array(results[head]), axis=0)
        avg_results.append(avg_result)

        avg_exists_result = np.mean(np.array(exists_results[head]), axis=0)
        exists_avg_results.append(avg_exists_result)

        avg_not_exists_result = np.mean(np.array(not_exists_results[head]), axis=0)
        not_exists_avg_results.append(avg_not_exists_result)

    return (
        np.array(avg_results),
        np.array(exists_avg_results),
        np.array(not_exists_avg_results),
    )


def load_head_decomp_by_rank(
    decomp_results_path,
    labels_path,
    perturb_type,
    score_df,
    heads,
    rank_start,
    rank_end,
    full_q_dict,
    selected_terms_dict,
    tokenizer,
):
    """Load head attention patterns for a specific rank range."""
    # Get query, doc ids
    filtered_scores = score_df[
        score_df["og_rank"].isin(list(range(rank_start, rank_end + 1)))
    ]
    target_docs = list(filtered_scores[["qid", "doc_id"]].to_records(index=False))

    results = []
    exists_results = []
    not_exists_results = []
    for qid, doc_id in target_docs:
        # Get label and segment
        label_fname = os.path.join(labels_path, "{}_{}_labels.txt".format(qid, doc_id))
        label = load_label_file(label_fname)

        (
            cls_idx,
            inj_idxs,
            q_term_inj_idxs,
            q_term_non_inj_idxs,
            non_q_term_idxs,
            sep_idx,  # ,
        ) = segment_tokens(
            label,
            perturb_type,
            str(qid),
            str(doc_id),
            full_q_dict,
            selected_terms_dict,
            tokenizer,
        )

        # Get results for each head and segment
        fname = os.path.join(
            decomp_results_path, "{}_{}_head_by_pos.npy".format(qid, doc_id)
        )
        data = np.load(fname)[0, :, :]

        cls = data[:, cls_idx]
        inj = np.mean(data[:, inj_idxs], axis=1)
        if q_term_inj_idxs:
            q_term_inj = np.mean(data[:, q_term_inj_idxs], axis=1)
        else:
            q_term_inj = np.zeros((4,))
        if q_term_non_inj_idxs:
            q_term_non_inj = np.mean(data[:, q_term_non_inj_idxs], axis=1)
        else:
            q_term_non_inj = np.zeros((4,))
        non_q_term = np.mean(data[:, non_q_term_idxs], axis=1)
        sep = data[:, sep_idx]

        result = [cls, inj, q_term_inj, q_term_non_inj, non_q_term, sep]
        results.append(result)

        # add to results
        if not q_term_inj_idxs:
            not_exists_results.append(result)
        else:
            exists_results.append(result)

    return (
        np.mean(np.array(results), axis=0),
        np.mean(np.array(exists_results), axis=0),
        np.mean(np.array(not_exists_results), axis=0),
    )


def load_all_results(result_type, scores_fname, results_path, fname_list=None):
    """Loads either block or head activation patching results for all queries."""
    if fname_list:
        with open(fname_list, "r") as f:
            fnames = [line.strip().split(".") for line in f.readlines()]
            fnames_head_pos = [fname[0] + "_by_pos." + fname[1] for fname in fnames]
        matching_files = [
            os.path.join(results_path, fname) for fname in fnames_head_pos
        ]
    else:
        fname_pattern = "*_{}.npy".format(result_type)
        matching_files = glob.glob(os.path.join(results_path, fname_pattern))

    # FILTER qid-doc combinations where p_score < og_score
    # if args.dataset == "TFC2":
    #     scores = pd.read_csv(scores_fname, usecols=["qid", "doc_id", "score_diff", "og_rank"])
    #     scores = scores[scores["score_diff"] > 0]
    #     # remove qid doc combinations from matching files that are not in scores
    #     matching_files = [file for file in matching_files if (int(file.split("\\")[-1].split("_")[0]), int(file.split("\\")[-1].split("_")[1])) in list(zip(scores["qid"], scores["doc_id"]))]

    # result per query: list of 100 np.array(n_layers, doc_len)
    # all results: n_queries length list of [100 * np.array(n_layers, doc_len)]

    # TODO: group by query? --> actually doesn't matter as long as label is correct? might care later though?
    results, labels, qids, doc_ids = [], [], [], []
    for file_path in matching_files:
        if not os.path.exists(file_path):
            print("file doesn't exist:", file_path)
            continue

        result_data = np.load(file_path)
        results.append(result_data)

        # Load labels for blocks
        split = file_path.split("\\")[-1].split("_")
        query_id = split[0]
        document_id = split[1]
        doc_ids.append(document_id)
        label_file_pattern = "{}_{}_labels.txt".format(query_id, document_id)
        label_files = glob.glob(os.path.join(results_path, label_file_pattern))

        if label_files:
            label_file = label_files[0]
            label_data = load_label_file(label_file)
            labels.append(label_data)
        else:
            labels.append(None)  # maintain order

        qids.append(query_id)

    return results, labels, qids, doc_ids


def load_queries(query_path, selected_term_path):
    """Returns dictionaries of query ids with their corresponding full query and selected query term."""
    query_dict = load_jsonl_file_into_dict(query_path)
    terms_df = pd.read_csv(selected_term_path, index_col="query_id")[
        "query_term"
    ].to_dict()

    # convert keys to strings
    terms_dict = {}
    for key, val in terms_df.items():
        terms_dict[str(key)] = val

    return query_dict, terms_dict


def get_query_data(query_id, query_dict, terms_dict):
    """Returns the full query and selected term."""
    return query_dict[query_id], terms_dict[query_id]


def segment_tokens_all(
    data, labels, qids, perturb_type, full_q_dict, selected_terms_dict, tokenizer, args
):
    """
    For block results, reformat into smaller segments.
    - Segment: [CLS] + [injected tokens] + [existing query term tokens matching injected tokens] + [existing qterm tokens not matching injected tokens] + [non query term tokens] + [SEP]
    """
    segmented_data = []
    qid_lookup = {}
    for i, result in enumerate(data):
        label = labels[i]
        qid = qids[i]

        if qid not in qid_lookup:
            full_q, selected_term = get_query_data(
                str(qid), full_q_dict, selected_terms_dict
            )
            qid_lookup[qid] = {
                "full_query_toks": [
                    tokenizer.tokenize(term) for term in full_q.split()
                ],
                "selected_term_toks": tokenizer.tokenize(selected_term),
            }

        selected_term_toks = qid_lookup[qid]["selected_term_toks"]
        full_q_tok_list = qid_lookup[qid]["full_query_toks"]

        # Get offset to find where original document starts and ends
        if perturb_type == "prepend":  # prepend
            og_doc_start_idx = len(selected_term_toks) + 1
        elif perturb_type == "append":  # append
            og_doc_start_idx = 1

        # Get [CLS] token
        cls_tok = result[:, :, 0][
            :, :, np.newaxis
        ]  # shape: (n_components, n_layers, 1)

        # segmentation for TFC1-I
        if args.dataset == "TFC1-I":
            # Get injected tokens
            if perturb_type == "prepend":  # prepend
                og_doc_end_idx = len(label) - 1
                inj_toks = np.mean(result[:, :, 1:og_doc_start_idx], axis=-1)[
                    :, :, np.newaxis
                ]  # shape: (n_components, n_layers, 1)
            elif perturb_type == "append":  # append
                og_doc_end_idx = len(label) - len(selected_term_toks) - 1
                inj_toks = np.mean(
                    result[:, :, -len(selected_term_toks) - 1 : -1], axis=-1
                )[:, :, np.newaxis]

            # Get all query term tokens existing in original document
            q_term_inj_idxs = []
            q_term_non_inj_idxs = []
            for query_tokens in full_q_tok_list:
                inj_tok_idxs = []
                non_inj_tok_idxs = []

                # strict query term matching
                for i in range(
                    og_doc_start_idx, og_doc_end_idx + 1 - len(query_tokens)
                ):
                    window = label[i : i + len(query_tokens)]
                    if window == query_tokens:
                        window_range = range(i, i + len(query_tokens))

                        if window == selected_term_toks:
                            inj_tok_idxs = inj_tok_idxs + [*window_range]
                        else:
                            non_inj_tok_idxs = non_inj_tok_idxs + [*window_range]

                q_term_inj_idxs = q_term_inj_idxs + inj_tok_idxs
                q_term_non_inj_idxs = q_term_non_inj_idxs + non_inj_tok_idxs

            if not q_term_inj_idxs:
                q_term_inj_toks = np.zeros((result.shape[0], result.shape[1], 1))
            else:
                q_term_inj_toks = np.mean(result[:, :, q_term_inj_idxs], axis=-1)[
                    :, :, np.newaxis
                ]  # shape: (n_components, n_layers, 1)

            if not q_term_non_inj_idxs:
                q_term_non_inj_toks = np.zeros((result.shape[0], result.shape[1], 1))
            else:
                q_term_non_inj_toks = np.mean(
                    result[:, :, q_term_non_inj_idxs], axis=-1
                )[:, :, np.newaxis]

            # Calculate all non query term tokens
            all_doc_idxs = list(range(og_doc_start_idx, len(label)))
            non_q_term_idxs = list(
                set(all_doc_idxs)
                - set(q_term_inj_idxs)
                - set(q_term_non_inj_idxs)
                - set([len(label) - 1])
            )
            non_q_term_toks = np.mean(result[:, :, non_q_term_idxs], axis=-1)[
                :, :, np.newaxis
            ]  # shape: (n_components, n_layers, 1)

        # segmentation for TFC1-R
        elif args.dataset == "TFC1-R":
            # define borders
            og_doc_end_idx = len(label)

            # Get all query term tokens existing in original document
            q_term_inj_idxs = []
            q_term_non_inj_idxs = []

            for query_tokens in full_q_tok_list:
                inj_tok_idxs = []
                non_inj_tok_idxs = []

                # strict query term matching
                for i in range(
                    og_doc_start_idx, og_doc_end_idx - len(selected_term_toks) + 1
                ):
                    window = label[i : i + len(query_tokens)]
                    if window == query_tokens:
                        window_range = range(i, i + len(query_tokens))

                        if window == selected_term_toks:
                            inj_tok_idxs = inj_tok_idxs + [*window_range]
                        else:
                            non_inj_tok_idxs = non_inj_tok_idxs + [*window_range]

                q_term_inj_idxs = q_term_inj_idxs + inj_tok_idxs
                q_term_non_inj_idxs = q_term_non_inj_idxs + non_inj_tok_idxs

            # all occurences of the selected query term are injected
            if not q_term_inj_idxs:
                q_term_inj_toks = np.zeros((result.shape[0], result.shape[1], 1))
                inj_toks = np.zeros((result.shape[0], result.shape[1], 1))
            else:
                inj_toks = np.mean(result[:, :, q_term_inj_idxs], axis=-1)[
                    :, :, np.newaxis
                ]  # shape: (n_components, n_layers, 1)
                q_term_inj_toks = np.zeros((result.shape[0], result.shape[1], 1))

            if not q_term_non_inj_idxs:
                q_term_non_inj_toks = np.zeros((result.shape[0], result.shape[1], 1))
            else:
                q_term_non_inj_toks = np.mean(
                    result[:, :, q_term_non_inj_idxs], axis=-1
                )[:, :, np.newaxis]

            # Calculate all non query term tokens
            all_doc_idxs = list(range(og_doc_start_idx, len(label)))
            non_q_term_idxs = list(
                set(all_doc_idxs)
                - set(q_term_non_inj_idxs)
                - set(q_term_inj_idxs)
                - set([len(label) - 1])
            )
            non_q_term_toks = np.mean(result[:, :, non_q_term_idxs], axis=-1)[
                :, :, np.newaxis
            ]  # shape: (n_components, n_layers, 1)

        # segmentation for TFC2
        elif args.dataset == "TFC2":
            # define borders
            og_doc_end_idx = (
                len(label)
                - (args.TFC2_n_filler - args.TFC2_K - 1)
                - ((args.TFC2_K + 1) * len(selected_term_toks))
                - 1
            )
            inj_toks_start_idx = og_doc_end_idx
            inj_toks_end_idx = len(label) - (args.TFC2_n_filler - args.TFC2_K - 1) - 1

            # calculate injected tokens - NOTE: klopt dit
            inj_toks = np.mean(
                result[:, :, inj_toks_start_idx:inj_toks_end_idx], axis=-1
            )[:, :, np.newaxis]

            # Get all query term tokens existing in original document
            q_term_inj_idxs = []
            q_term_non_inj_idxs = []

            for query_tokens in full_q_tok_list:
                inj_tok_idxs = []
                non_inj_tok_idxs = []

                # strict query term matching
                for i in range(
                    og_doc_start_idx, og_doc_end_idx - len(selected_term_toks) + 1
                ):
                    window = label[i : i + len(query_tokens)]
                    if window == query_tokens:
                        window_range = range(i, i + len(query_tokens))

                        if window == selected_term_toks:
                            inj_tok_idxs = inj_tok_idxs + [*window_range]
                        else:
                            non_inj_tok_idxs = non_inj_tok_idxs + [*window_range]

                q_term_inj_idxs = q_term_inj_idxs + inj_tok_idxs
                q_term_non_inj_idxs = q_term_non_inj_idxs + non_inj_tok_idxs

            if not q_term_inj_idxs:
                q_term_inj_toks = np.zeros((result.shape[0], result.shape[1], 1))
            else:
                q_term_inj_toks = np.mean(result[:, :, q_term_inj_idxs], axis=-1)[
                    :, :, np.newaxis
                ]  # shape: (n_components, n_layers, 1)

            if not q_term_non_inj_idxs:
                q_term_non_inj_toks = np.zeros((result.shape[0], result.shape[1], 1))
            else:
                q_term_non_inj_toks = np.mean(
                    result[:, :, q_term_non_inj_idxs], axis=-1
                )[:, :, np.newaxis]

            # Calculate all non query term tokens
            all_doc_idxs = list(range(og_doc_start_idx, len(label)))
            inj_toks_idxs = list(range(inj_toks_start_idx, inj_toks_end_idx + 1))
            non_q_term_idxs = list(
                set(all_doc_idxs)
                - set(q_term_inj_idxs)
                - set(q_term_non_inj_idxs)
                - set(inj_toks_idxs)
            )
            non_q_term_toks = np.mean(result[:, :, non_q_term_idxs], axis=-1)[
                :, :, np.newaxis
            ]  # shape: (n_components, n_layers, 1)

        # Get [SEP] token
        sep_tok = result[:, :, -1][:, :, np.newaxis]

        # Concatenate along the last axis
        new_result = np.concatenate(
            [
                cls_tok,
                inj_toks,
                q_term_inj_toks,
                q_term_non_inj_toks,
                non_q_term_toks,
                sep_tok,
            ],
            axis=2,
        )

        segmented_data.append(new_result)

    # NOTE: not the nicest solution, but needed as there are unexpected infs and nans (not many)
    print(
        f"Existence of NaNs: {np.isnan(segmented_data).sum()} and infs: {np.isinf(segmented_data).sum()}"
    )
    segmented_data = np.where(np.isinf(segmented_data), np.nan, segmented_data)
    return np.nanmean(np.array(segmented_data), axis=0)


def plot_blocks_plotly(data, labels, save_path, args):
    """Plot block experiment results"""

    fig = sp.make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Residual Stream", "Attn Output", "MLP Output"],
        shared_yaxes=True,
        horizontal_spacing=0.1,
    )

    # Create heatmaps for each experiment
    for i in range(3):
        heatmap_data = data[i, :, :]
        heatmap = go.Heatmap(
            z=heatmap_data, colorscale="RdBu", zmin=-1, zmax=1
        )  # zmin=-1, zmax=1
        fig.add_trace(heatmap, row=1, col=i + 1)

    if args.dataset == "TFC1-I":
        suffix = f" {args.TFC1_I_perturb_type}"
    elif args.dataset == "TFC1-R":
        suffix = ""
    elif args.dataset == "TFC2":
        suffix = f", K = {args.TFC2_K}"

    fig.update_layout(
        title=f"Activation Patching Per Block for {args.dataset}{suffix}",
        xaxis=dict(
            showline=True,
            showgrid=False,
            tickvals=np.arange(len(labels)),
            ticktext=labels,
        ),
        yaxis=dict(title="Layer", showline=True, showgrid=False),
        xaxis2=dict(
            showline=True,
            showgrid=False,
            tickvals=np.arange(len(labels)),
            ticktext=labels,
        ),
        xaxis3=dict(
            showline=True,
            showgrid=False,
            tickvals=np.arange(len(labels)),
            ticktext=labels,
        ),
        width=1000,
    )

    # rotate x labels vertical and make the labels a bit bigger
    for i in range(3):
        fig.update_xaxes(
            tickangle=90, title_font_size=20, tickfont=dict(size=20), row=1, col=i + 1
        )
        fig.update_yaxes(tickfont=dict(size=20), title_font_size=16, row=1, col=i + 1)

    # make title bigger
    fig.update_layout(title_font_size=20)

    if save_path:
        fig.write_image(save_path, format="png", width=800, height=400, scale=2)

    return fig


def plot_heads(data, save_path, args, topbottom):
    """Plot head experiment results"""

    # Get max absolute value for color normalization
    if args.dataset == "TFC1-I":
        max_abs_val = 1
    else:
        max_abs_val = np.max(np.abs(data))

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        data,
        cmap="RdBu",
        vmin=-max_abs_val,
        vmax=max_abs_val,
        xticklabels=True,
        yticklabels=True,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 11}
    )

    if args.dataset == "TFC1-I":
        suffix = f"{args.dataset} {args.TFC1_I_perturb_type}"
    elif args.dataset == "TFC2":
        suffix = f"{args.dataset}-K{args.TFC2_K}"
    plt.title(
        f"{suffix}: {topbottom}",
        fontsize=20,
    )
    plt.xlabel("Head", fontsize=16)
    plt.ylabel("Layer", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)


    if save_path:
        plt.savefig(save_path, dpi=1200)
        plt.close()

    return plt.gcf()


def grouped_bar_chart(data_array, heads, label_segs, save_path=None):
    # Create a DataFrame
    df = pd.DataFrame(data_array.T, columns=[f"L{head[0]}H{head[1]}" for head in heads])
    df["Dimension"] = label_segs
    df_melted = pd.melt(df, id_vars="Dimension", var_name="Head", value_name="Values")

    # Create the grouped bar chart with seaborn
    sns.set(style="whitegrid")
    ax = sns.barplot(
        x="Dimension", y="Values", hue="Head", data=df_melted, palette="Set3"
    )

    # Add labels and title
    ax.set(
        xlabel="Token Type",
        ylabel="Avg Attention Scores",
        title="Top Ranked Perturbed Documents",
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, ha="center")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=1200)
        plt.close()

    # Return the matplotlib figure
    return plt.gcf()


def main(args):
    # setup
    perturb_type = args.TFC1_I_perturb_type
    plot = [args.experiment_type]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(f"figures/{args.dataset}", exist_ok=True)
    labels = [
        "$\\text{tok}_{\\text{CLS}}$",
        "$\\text{tok}_{\\text{inj}}$",
        "$\\text{tok}_{\\text{qterm+}}$",
        "$\\text{tok}_{\\text{qterm-}}$",
        "$\\text{tok}_{\\text{other}}$",
        "$\\text{tok}_{\\text{SEP}}$",
    ]
    if args.dataset == "TFC2":
        scores_csv_path = f"data/{args.dataset}/computed_results_{args.TFC1_I_perturb_type}_{args.TFC2_K}.csv"
        folder_suffix = f"/{args.TFC1_I_perturb_type}/{args.TFC2_K}"
    else:
        scores_csv_path = (
            f"data/{args.dataset}/computed_results_{args.TFC1_I_perturb_type}.csv"
        )
        folder_suffix = f"/{args.TFC1_I_perturb_type}/"

    # Load model and tokenizer
    print("loading tokenizer")
    hf_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    # Load full queries and selected terms
    full_query_dict, selected_terms_dict = load_queries(
        "data/queries.jsonl", f"data/selected_query_terms.csv"
    )

    print(f"| Plotting {plot} for {args.dataset} dataset")

    if "block" in plot:
        # Load, segment, and plot average block results
        all_block_results, all_block_labels, all_block_qids, _ = load_all_results(
            "block", scores_csv_path, f"{args.results_folder}/results/{folder_suffix}"
        )
        all_block_segmented = segment_tokens_all(
            all_block_results,
            all_block_labels,
            all_block_qids,
            perturb_type,
            full_query_dict,
            selected_terms_dict,
            tokenizer,
            args,
        )
        _ = plot_blocks_plotly(
            all_block_segmented,
            labels,
            f"{args.figure_folder}/normed_all_block_seg.png",
            args,
        )

    if "head_all" in plot:
        #  Load and plot head results for top/bottom ranked documents
        print("plotting head results for top ranked documents")
        top_ranked_doc_head_results, _, _ = load_doc_results_by_rank(
            scores_csv_path,
            "head",
            f"{args.results_folder}/results_head_decomp/{folder_suffix}",
            f"{args.results_folder}/results_head_decomp/{folder_suffix}",
            args,
            rank_start=1,
            rank_end=10,
        )

        avg_top_ranked_doc_head_results = np.mean(top_ranked_doc_head_results, axis=0)
        _ = plot_heads(
            avg_top_ranked_doc_head_results,
            f"{args.figure_folder}/FILTERED_top_ranked_doc_head_results.png",
            args,
            "top",
        )

        print("plotting head results for bottom ranked documents")
        bottom_ranked_doc_head_results, _, _ = load_doc_results_by_rank(
            scores_csv_path,
            "head",
            f"{args.results_folder}/results_head_decomp/{folder_suffix}",
            f"{args.results_folder}/results_head_decomp/{folder_suffix}",
            args,
            rank_start=91,
            rank_end=100,
        )
        avg_bottom_ranked_doc_head_results = np.mean(
            bottom_ranked_doc_head_results, axis=0
        )
        _ = plot_heads(
            avg_bottom_ranked_doc_head_results,
            f"{args.figure_folder}/FILTERED_bottom_ranked_doc_head_results.png",
            args,
            "bottom",
        )

    if "head_pos" in plot:
        # head patching and attention patterns for top ranked documents by position
        # NOTE: these are the old heads. Other heads come up in the reproduction
        # heads = [(0,9), (1,6), (2,3), (3,8)]
        heads = [(0, 9), (1, 6), (2, 3)]
        scores_df = pd.read_csv(scores_csv_path)
        head_decomp_data = load_head_decomp_by_rank(
            f"{args.results_folder}/results_head_decomp/{folder_suffix}",
            f"{args.results_folder}/results/{folder_suffix}",
            perturb_type,
            scores_df,
            heads,
            1,
            10,
            full_query_dict,
            selected_terms_dict,
            tokenizer,
        )
        labels = [
            "$\\text{tok}_{\\text{CLS}}$",
            "$\\text{tok}_{\\text{inj}}$",
            "$\\text{tok}_{\\text{qterm+}}$",
            "$\\text{tok}_{\\text{qterm-}}$",
            "$\\text{tok}_{\\text{other}}$",
            "$\\text{tok}_{\\text{SEP}}$",
        ]
        _ = grouped_bar_chart(
            head_decomp_data[0].T,
            heads,
            labels,
            save_path=f"{args.figure_folder}/head_decomp_top_heads_overall.png",
        )
        _ = grouped_bar_chart(
            head_decomp_data[1].T,
            heads,
            labels,
            save_path=f"{args.figure_folder}/head_decomp_top_heads_exists.png",
        )
        _ = grouped_bar_chart(
            head_decomp_data[2].T,
            heads,
            labels,
            save_path=f"{args.figure_folder}/head_decomp_top_heads_not_exists.png",
        )

    if "head_attn" in plot:
        # NOTE: these are the old heads. Other heads come up in the reproduction
        # heads = [(0,9), (1,6), (2,3), (3,8)]
        heads = [(0, 9), (1, 6), (2, 3)]
        scores_df = pd.read_csv(scores_csv_path)
        head_data = load_head_pattern_by_rank(
            f"{args.results_folder}/results_attn_pattern/{folder_suffix}",
            f"{args.results_folder}/results/{folder_suffix}",
            perturb_type,
            scores_df,
            heads,
            1,
            10,
            full_query_dict,
            selected_terms_dict,
            tokenizer,
        )
        labels = [
            r"$\text{tok}_{\text{inj}}$" + "\n"
            + r"$\text{to}$" + "\n"
            + r"$\text{tok}_{\text{qterm+}}$",
            r"$\text{tok}_{\text{qterm+}}$" + "\n"
            + r"$\text{to}$" + "\n"
            + r"$\text{tok}_{\text{inj+}}$",
            r"$\text{tok}_{\text{inj}}$" + "\n"
            + r"$\text{to}$" + "\n"
            + r"$\text{tok}_{\text{qterm-}}$",
            r"$\text{tok}_{\text{inj}}$" + "\n"
            + r"$\text{to}$" + "\n"
            + r"$\text{tok}_{\text{SEP}}$",
            r"$\text{tok}_{\text{other}}$" + "\n"
            + r"$\text{to}$" + "\n"
            + r"$\text{tok}_{\text{SEP}}$",
        ]

        _ = grouped_bar_chart(
            head_data[0],
            heads,
            labels,
            save_path=f"{args.figure_folder}/head_attn_pattern_{perturb_type}_overall.png",
        )
        _ = grouped_bar_chart(
            head_data[1],
            heads,
            labels,
            save_path=f"{args.figure_folder}/head_attn_pattern_{perturb_type}_exists.png",
        )
        _ = grouped_bar_chart(
            head_data[2],
            heads,
            labels,
            save_path=f"{args.figure_folder}/head_attn_pattern_{perturb_type}_not_exists.png",
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results.")
    parser.add_argument(
        "--dataset", default="TFC2", choices=["TFC1-I", "TFC1-R", "TFC2"]
    )
    parser.add_argument(
        "--experiment_type",
        default="block",
        choices=["block", "head_all", "head_all", "head_pos", "head_attn", "labels"],
        help="What will be patched (e.g., block).",
    )
    parser.add_argument(
        "--TFC1_I_perturb_type",
        default="append",
        choices=["append", "prepend"],
        help="The perturbation to apply (e.g., append).",
    )
    parser.add_argument(
        "--TFC2_K",
        default=1,
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50],
        help="K",
    )
    parser.add_argument(
        "--filter",
        default=False,
        type=bool,
        help="For TFC2: whether to filter out documents where the perturbed score is lower than the original score.",
    )
    args = parser.parse_args()

    # non variable args
    if args.dataset == "TFC2":
        args.figure_folder = f"figures/{args.dataset}/{args.TFC2_K}"
    elif args.dataset == "TFC1-I":
        args.figure_folder = f"figures/{args.dataset}/{args.TFC1_I_perturb_type}"
    elif args.dataset == "TFC1-R":
        args.figure_folder = f"figures/{args.dataset}/"
    args.results_folder = f"results/{args.dataset}"
    os.makedirs(args.figure_folder, exist_ok=True)
    args.TFC2_n_filler = 51

    _ = main(args)
