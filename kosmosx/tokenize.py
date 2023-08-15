import argparse
import multiprocessing
from itertools import chain

from datasets import load_dataset

from kosmosx.model import KosmosTokenizer


class BuildDataset:
    def __init__(self, seed=42, seq_len=8192, hf_account="YOUR HUGGINGFACE API KEY", dataset_name="uggingFaceM4/VQAv2"):
        self.SEED = seed
        self.SEQ_LEN = seq_len
        self.NUM_CPU = multiprocessing.cpu_count()
        self.HF_ACCOUNT_REPO = hf_account
        self.DATASET_NAME = dataset_name
        self.tokenizer = KosmosTokenizer.tokenize

    def tokenize_function(self, example):
        return self.tokenizer([t + self.tokenizer.eos_token for t in example["text"]])

    def group_texts(self, examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= self.SEQ_LEN:
            total_length = (total_length // self.SEQ_LEN) * self.SEQ_LEN
        result = {
            k: [t[i : i + self.SEQ_LEN] for i in range(0, total_length, self.SEQ_LEN)]
            for k, t in concatenated_examples.items()
        }
        return result

    def build(self):
        train_dataset = load_dataset(self.DATASET_NAME, split="train", streaming=True)
        tokenized_dataset = train_dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.NUM_CPU,
            remove_columns=["text"],
        )
        train_tokenized_dataset = tokenized_dataset.map(
            self.group_texts,
            batched=True,
            num_proc=self.NUM_CPU,
        )
        train_tokenized_dataset.push_to_hub(self.HF_ACCOUNT_REPO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and push dataset to Hugging Face Hub")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--seq_len", type=int, default=8192, help="Sequence length for processing")
    parser.add_argument("--hf_account", type=str, default="YOUR HUGGINGFACE API KEY", help="Hugging Face account name and repo")
    parser.add_argument("--dataset_name", type=str, default="uggingFaceM4/VQAv2", help="Name of the dataset to process")
    args = parser.parse_args()
    dataset_builder = BuildDataset(seed=args.seed, seq_len=args.seq_len, hf_account=args.hf_account, dataset_name=args.dataset_name)
    dataset_builder.build()