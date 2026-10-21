# saves the TinyStories dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

tokenizer = AutoTokenizer.from_pretrained('tokenizers/notalovesong')

if __name__ == '__main__':

    def tokenize(examples):
        ids = tokenizer.encode(examples['text'], add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)

        return {'ids': ids, 'len': len(ids)}

    raw_datset = [
        ' '.join('This is not a love song.' for _ in range(100)) for _ in range(100)
    ]

    split = Dataset.from_dict({'text': raw_datset})

    dataset = DatasetDict({'train': split, 'validation': split})
    # tokenize the dataset
    tokenized = dataset.map(
        tokenize,
        remove_columns=['text'],
        desc='tokenizing the splits',
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
