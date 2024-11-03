# 🦛 Chonkie Docs

> UGH, do I _need_ to explain how to use Chonkie? Man, that's a bummer... To be honest, Chonkie is very easy, with little documentation necessary, but just in case, I'll include some here. 

# Table Of Contents

- [🦛 Chonkie Docs](#-chonkie-docs)
- [Table Of Contents](#table-of-contents)
- [Design CHONKosophy](#design-chonkosophy)
  - [Dependency Table](#dependency-table)
  - [Why Chunking is needed? (And may always be needed!)](#why-chunking-is-needed-and-may-always-be-needed)
- [Chunkers](#chunkers)
  - [TokenChunker](#tokenchunker)
    - [Initialization](#initialization)
    - [Methods](#methods)
    - [Example](#example)
  

# Design CHONKosophy

> Did you know that Hippos are surprisingly smart? 

A lot of thought went into this repository and I want to take some space here to fully go over some of the decisions made: the what, why and how? 

1. **Chonkie is very smart**
2. **Chonkie is surprisingly lightweight**
3. **Chonkie is superrrrr fast**



## Dependency Table

As per the details mentioned in the [Design](#design-chonkosophy) section, Chonkie is lightweight because it keeps most of the dependencies for each chunker seperate, making it more of an aggregate of multiple repositories and python packages. The optional dependencies feature in Python really helps with this. 

| Chunker  | Default | 'sentence' | 'semantic' | 'all' |
|----------|----------|----------|----------|----------|
| TokenChunker        |✅|✅|✅|✅|
| WordChunker         |✅|✅|✅|✅|
| SentenceChunker     |⚠️|✅|⚠️|✅|
| SemanticChunker     |❌|❌|⚠️/✅|✅|
| SPDMChunker         |❌|❌|⚠️/✅|✅|

Note: In the above table `⚠️/✅` meant that some features would be disabled but the Chunker would work nonetheless. 

What you could infer from the table is that, while it might be of inconvinience in the short-run to have it split like that, you can do surprisingly a lot with just the Defualt dependencies (which btw are super light). Furthermore, even our max dependencies option `all` is lightweight in comparison to some of the other libraries that one might use for such tasks. 


## Why Chunking is needed? (And may always be needed!)

While you might be aware of models having longer and longer contexts in recent times (as of 2024), models have yet to reach the stage where adding additional context to them comes for free. Additional context, even with the greatest of model architectures comes at a O(n) penalty in speed, to say nothing of the additional memory requirements. And as long as we belive in that attention is all we need, it doesn't seem likely we would be free from this penalty. 

That means, to make models run efficiently (lower speed, memory) it is absoulutely vital that we provide the most accurate information it needs during the retrieval phase. 

Accuracy is one part during retrieval and the other is granularity. You might be able to extract the relevant Article out for model to work with, but if only 1 line is relevant from that passage, you are in effect adding a lot of noise that would hamper and confuse the model in practice. You want and hope to give the model only what it should require ideally (of course, the ideal scenario is rarely ever possible). This finally brings us to granularity and retrieval accuracy. 

Representation models (or embedding models as you may call them) are great at representing large amount of information (sometimes pages of text) in a limited space of just 700-1000 floats, but that doesn't mean it does not suffer from any loss. Most representation is lossy, and if we have many concepts being covered in the same space, it is often that much of it would be lost. However, singluar concepts and explainations breed stronger representation vectors. It then becomes vital again to make sure we don't dilute the representation with noise. 

All this brings me back to chunking. Chunking, done well, can make sure your representation vector (or embedding) is of high-quality to be able to retrieve the best context for your model to generate with. And that in turn, leads to better quality RAG generations. Therefore, I believe chunking is here to stay as long as RAG is here. And hence, it becomes important that we give it little more than a after-thought. 

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