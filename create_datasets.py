# imports
import json
import pandas as pd
import random
import string
import argparse


# setup
FILLER = "a"


# functions
def load_json_to_df(file_path):
    """Load a nested json file into a pandas DataFrame."""
    # open json
    with open(file_path, "r") as f:
        data = json.load(f)

    # create a list to store the flattened data
    flattened_data = []

    # iterate through the nested structure
    for outer_key, inner_dict in data["corpus"].items():
        for inner_key, content in inner_dict.items():
            # Create a row with all the information
            row = {
                "query_id": outer_key,
                "doc_id": inner_key,
                "title": content.get("title", ""),
                "text": content.get("text", ""),
                "query_term_orignal_ct": content.get("query_term_orignal_ct", 0),
            }
            flattened_data.append(row)

    # convert to DataFrame
    df = pd.DataFrame(flattened_data)
    return df


def save_dataset_as_dict(corpus):
    """Convert the corpus to a nested dictionary."""
    corpus_dict = {}
    for _, row in corpus.iterrows():
        query_id = row["query_id"]
        doc_id = row["doc_id"]
        title = row["title"]
        text = row["text"]
        query_term_orignal_ct = row["query_term_orignal_ct"]
        query_term = row["query_term"]

        if query_id not in corpus_dict:
            corpus_dict[query_id] = {}

        corpus_dict[query_id][doc_id] = {
            "title": title,
            "text": text,
            "query_term_orignal_ct": query_term_orignal_ct,
            "query_term": query_term,
        }

    return corpus_dict


def save_dataset(corpus, baseline_corpus, args, **kwargs):
    """Save the perturbed and baseline datasets as json files."""
    # convert to nested dict
    corpus_dict = save_dataset_as_dict(corpus)
    baseline_corpus_dict = save_dataset_as_dict(baseline_corpus)

    # define path to save
    if args.experiment == "TFC1-I":
        perturbed_path = f"data/{args.experiment}/{args.experiment}_{args.TFC1_I_perturb_type}_corpus.json"
        baseline_path = f"data/{args.experiment}/{args.experiment}_{args.TFC1_I_perturb_type}_baseline.json"
    elif args.experiment == "TFC2":
        perturbed_path = (
            f'data/{args.experiment}/{args.experiment}_{kwargs["K"]}_corpus.json'
        )
        baseline_path = (
            f'data/{args.experiment}/{args.experiment}_{kwargs["K"]}_baseline.json'
        )
    else:
        perturbed_path = f"data/{args.experiment}/{args.experiment}_corpus.json"
        baseline_path = f"data/{args.experiment}/{args.experiment}_baseline.json"

    # save to json
    with open(perturbed_path, "w") as f:
        json.dump({"corpus": corpus_dict}, f, indent=4)
    with open(baseline_path, "w") as f:
        json.dump({"corpus": baseline_corpus_dict}, f, indent=4)

    return


def perturb_dataset(args):
    """Perturb the baseline dataset according to the specified experiment."""
    # convert to a pandas df
    baseline_corpus_df = load_json_to_df("data/baseline.json")

    # load queries
    queries = pd.read_csv(
        "data/QIDs_with_text.csv",
        header=None,
        names=["query_id", "query_text"],
        dtype={"query_id": str, "query_text": str},
    )

    # load randomly sampled query terms
    query_term = pd.read_csv(
        "data/selected_query_terms.csv",
        header=None,
        names=["query_id", "query_term"],
        dtype={"query_id": str, "query_term": str},
    )

    # merge queries with corpus
    queries = pd.merge(
        queries, query_term, left_on="query_id", right_on="query_id", how="left"
    )
    corpus = baseline_corpus_df.merge(
        queries, left_on="query_id", right_on="query_id", how="left"
    )
    print(f"Loaded {len(corpus)} documents., Columns are {corpus.columns}")
    baseline_corpus = corpus.copy()

    # TFC1-I: append or prepend query term
    if args.experiment == "TFC1-I":
        for i, row in corpus.iterrows():
            text = row["text"]
            query_term = row["query_term"]

            if args.TFC1_I_perturb_type == "append":
                perturbed_text = text + " " + query_term
                baseline_perturbed_text = text + " " + FILLER
            elif args.TFC1_I_perturb_type == "prepend":
                perturbed_text = query_term + " " + text
                baseline_perturbed_text = FILLER + " " + text

            corpus.at[i, "text"] = perturbed_text
            baseline_corpus.at[i, "text"] = baseline_perturbed_text

        save_dataset(corpus, baseline_corpus, args)

    # TFC1-R: replace query term with random filler
    elif args.experiment == "TFC1-R":
        for i, row in corpus.iterrows():
            text = row["text"]
            query_term = row["query_term"]
            perturbed_text = ""

            for word in text.split():
                # strip punctuation
                check_word = word.strip().lower()
                check_word = check_word.translate(
                    str.maketrans("", "", string.punctuation)
                )

                # replace query term for random filler
                if check_word == query_term:
                    perturbed_text += FILLER + " "
                else:
                    perturbed_text += word + " "

            corpus.at[i, "text"] = perturbed_text

        save_dataset(corpus, baseline_corpus, args)

    # TFC2: add query term K times
    elif args.experiment == "TFC2":
        for K in args.TFC2_K:
            if K > args.TFC2_maxK:
                print(f"Skipping K={K} as it is greater than max K = {args.TFC2_maxK}")

            TFC2_corpus = corpus.copy()
            TFC2_baseline_corpus = baseline_corpus.copy()
            for i, row in TFC2_corpus.iterrows():
                text = row["text"] + " "
                query_term = row["query_term"]
                perturbed_text = (
                    text
                    + " ".join([query_term] * (K + 1))
                    + " "
                    + " ".join([FILLER] * (args.TFC2_maxK - K))
                )
                baseline_perturbed_text = (
                    text
                    + " ".join([query_term] * K)
                    + " "
                    + " ".join([FILLER] * (args.TFC2_maxK - K + 1))
                )

                TFC2_corpus.at[i, "text"] = perturbed_text
                TFC2_baseline_corpus.at[i, "text"] = baseline_perturbed_text

            save_dataset(TFC2_corpus, TFC2_baseline_corpus, args, K=K)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run activation patching with the specific patching experiment and perturbation types."
    )
    parser.add_argument(
        "--experiment",
        default="TFC2",
        choices=["TFC1-I", "TFC1-R", "TFC2"],
        help="The perturbation to apply (e.g., append).",
    )
    parser.add_argument(
        "--TFC1_I_perturb_type",
        default="append",
        choices=["append", "prepend"],
        help="Wether to add the query term at the beginning or end of the text.",
    )
    parser.add_argument(
        "--TFC2_K",
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50],
        type=int,
        nargs="+",
        help="The number of the selected query term to add to the baseline. K + 1 will be added to the perturbed doc",
    )
    args = parser.parse_args()
    perturb_dataset(args)
