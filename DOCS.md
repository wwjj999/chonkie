# 🦛 Chonkie Docs

> UGH, do I _need_ to explain how to use Chonkie? Man, that's a bummer... To be honest, Chonkie is very easy, with little documentation necessary, but just in case, I'll include some here. 

# Table Of Contents

- [🦛 Chonkie Docs](#-chonkie-docs)
- [Table Of Contents](#table-of-contents)
- [Chunkers](#chunkers)
  - [TokenChunker](#tokenchunker)
    - [Initialization](#initialization)
    - [Methods](#methods)
    - [Example](#example)

# Chunkers
## TokenChunker

The `TokenChunker` class is designed to split text into overlapping chunks based on a specified token size. This is particularly useful for applications that need to process text in manageable pieces while maintaining some context between chunks.

### Initialization

To initialize a `TokenChunker`, you need to provide a tokenizer, the maximum number of tokens per chunk, and the number of tokens to overlap between chunks.

```python
from tokenizers import Tokenizer
from chonkie.chunker import TokenChunker

# Initialize the tokenizer (example using GPT-2 tokenizer)
tokenizer = Tokenizer.from_pretrained("gpt2")

# Initialize the TokenChunker
chunker = TokenChunker(tokenizer=tokenizer, chunk_size=512, chunk_overlap=128)
```

### Methods

`chunk(text: str) -> List[Chunk]`
This method splits the input text into overlapping chunks of the specified token size.

`Args:`
* `text` (str): The input text to be chunked.
`Returns:`
* `List[Chunk]`: A list of Chunk objects containing the chunked text and metadata.

### Example

```python
text = "Your input text here."
chunks = chunker.chunk(text)

for chunk in chunks:
    print(f"Chunk text: {chunk.text}")
    print(f"Start index: {chunk.start_index}")
    print(f"End index: {chunk.end_index}")
    print(f"Token count: {chunk.token_count}")
```