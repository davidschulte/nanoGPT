from transformers import AutoTokenizer
from datasets import load_dataset

if __name__ == '__main__':
    raw_dataset = load_dataset(
        'BabyLM-community/BabyLM-2026-Strict-Small', split='train'
    )

    train_split = raw_dataset.train_test_split(0.2, seed=42)['train']

    def batch_iterator(batch_size=1000):
        for i in range(0, len(train_split), batch_size):
            yield train_split[i : i + batch_size]['text']

    old_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=2**13
    )
    new_tokenizer.eos_token = '<|endoftext|>'
    new_tokenizer.bos_token = '<|endoftext|>'

    new_tokenizer.save_pretrained('./tokenizers/babylm10m/')
    print(new_tokenizer.vocab)
