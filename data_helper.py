import torch
import os
import pickle
import logging
from torch.utils.data import Dataset
from transformers import DataCollator
from overrides import overrides
import csv


logger = logging.getLogger(__name__)

labels_to_ids = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "like": 4,
    "sadness": 5,
    "surprise": 6
    }

class ExampleDataset(Dataset):
    def __init__(self, examples_dict):
        self.examples = examples_dict
    
    def __len__(self):
        return len(self.examples["input_ids"])
    
    def __getitem__(self, i):
        single_example = {}
        for k, v in self.examples.items():
            single_example.update({k: v[i]})
        return single_example


class DictDataCollator(DataCollator):
    @overrides
    def collate_batch(self, batch_dict_example):
        inputs = {}
        input_ids = []
        attention_mask = []
        token_type_ids = []
        position_ids = []
        label_ids = []

        max_len = -1
        for x in batch_dict_example:
            max_len = max(len(x["input_ids"]), max_len)

        # PADDING
        for dict_example in batch_dict_example:
            src_padding_len = max_len - len(dict_example["input_ids"])
            
            input_ids.append(dict_example["input_ids"] + [0] * src_padding_len)
            attention_mask.append(dict_example["attention_mask"] + [0] * src_padding_len)
            position_ids.append(dict_example["position_ids"] + \
                [x for x in range(len(dict_example["position_ids"]), max_len)]
                )
            token_type_ids.append(dict_example["token_type_ids"] + [0] * src_padding_len)
            label_ids.append(dict_example["label_ids"])
            
        inputs["input_ids"] = torch.tensor(input_ids, dtype=torch.long)
        inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.long)
        inputs["position_ids"] = torch.tensor(position_ids, dtype=torch.long)
        inputs["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)
        inputs["labels"] = torch.tensor(label_ids, dtype=torch.long)

        return inputs

def get_examples(
    examples_path,
    tokenizer,
    max_src_len,
    data_mode="csv"
    ):
    """
        get examples for text format file
    """
    directory, filename = os.path.split(examples_path)
    cached_features_file = os.path.join(directory, 
        f"cached_src_{max_src_len}_" + filename + ".pkl"
        )
    if os.path.exists(cached_features_file):
        logger.info(f"Loading features from cached file {cached_features_file}.")
        with open(cached_features_file, "rb") as handle:
            examples = pickle.load(handle)
            return examples
    
    examples = {
        "input_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
        "label_ids": [],
        "position_ids": [],
        }
    if data_mode == "csv":
        with open(examples_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter='\t')
            for parts in reader:
                if len(parts) != 3:
                    continue
                text = parts[1]
                label = parts[2]
                tokens = tokenizer.tokenize("[CLS]" + text + "[SEP]") 
                input_ids = [tokenizer.convert_tokens_to_ids(word) for word in tokens]
                segment_ids = [0] * len(input_ids)
                position_ids = [x for x in range(len(input_ids))]
                input_mask = [1] * len(input_ids)
                label_ids = labels_to_ids[label]
                # left truncated
                if len(input_ids) > max_src_len:
                    input_ids = input_ids[:max_src_len]
                    segment_ids = segment_ids[:max_src_len]
                    input_mask = input_mask[:max_src_len]
                    position_ids = position_ids[:max_src_len]
                    if input_ids[-1] != tokenizer.sep_token_id:
                        input_ids[-1] = tokenizer.sep_token_id
                
                examples["input_ids"].append(input_ids)
                examples["attention_mask"].append(input_mask)
                examples["token_type_ids"].append(segment_ids)
                examples["position_ids"].append(position_ids)
                examples["label_ids"].append(label_ids)


    logger.info(f"There are {len(examples['input_ids'])} exmaples in {filename}")
    with open(cached_features_file, "wb") as handle:
        pickle.dump(examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return examples


if __name__ == "__main__":
    a = torch.tensor([1,2,3,4,5])
    print(a)

