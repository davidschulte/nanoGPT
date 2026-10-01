from transformers import AutoTokenizer
from datasets import load_dataset

if __name__ == '__main__':
    raw_dataset = load_dataset('roneneldan/TinyStories', split='train')

    def batch_iterator(batch_size=1000):
        for i in range(0, len(raw_dataset), batch_size):
            yield raw_dataset[i : i + batch_size]['text']

    old_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    new_tokenizer = old_tokenizer.train_new_from_iterator(
        batch_iterator(), vocab_size=5000
    )
    new_tokenizer.eos_token = '<|endoftext|>'
    new_tokenizer.bos_token = '<|endoftext|>'

    new_tokenizer.save_pretrained('./tokenizers/tinystories')
