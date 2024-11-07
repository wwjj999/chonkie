# 🦛 Chonkie Docs

> UGH, do I _need_ to explain how to use Chonkie? Man, that's a bummer... To be honest, Chonkie is very easy, with little documentation necessary, but just in case, I'll include some here. 

# Table Of Contents

- [🦛 Chonkie Docs](#-chonkie-docs)
- [Table Of Contents](#table-of-contents)
- [Installation](#installation)
  - [Basic Installation](#basic-installation)
  - [Installation Options](#installation-options)
  - [Dependency Table](#dependency-table)
- [Quick Start](#quick-start)
- [Design CHONKosophy](#design-chonkosophy)
  - [Why Chunking is needed? (And may always be needed!)](#why-chunking-is-needed-and-may-always-be-needed)
  - [Does speed matter while Chunking? (TL;DR: YES!!!)](#does-speed-matter-while-chunking-tldr-yes)
- [Chunkers](#chunkers)
  - [TokenChunker](#tokenchunker)
  - [WordChunker](#wordchunker)
  - [SentenceChunker](#sentencechunker)
  - [SemanticChunker](#semanticchunker)
  - [SDPMChunker](#sdpmchunker)
- [API Reference](#api-reference)
  - [Chunk Object](#chunk-object)
  - [Sentence Chunk Object](#sentence-chunk-object)
  - [Semantic Chunk Object](#semantic-chunk-object)
  

# Installation

## Basic Installation
```bash
pip install chonkie
```

## Installation Options
Chonkie uses optional dependencies to keep the base installation lightweight. Choose the installation that best fits your needs:

| Installation Command | Use Case | Dependencies Added |
|---------------------|----------|-------------------|
| `pip install chonkie` | Basic token and word chunking | tokenizers |
| `pip install chonkie[sentence]` | Sentence-based chunking | + spacy |
| `pip install chonkie[semantic]` | Semantic chunking | + sentence-transformers, numpy |
| `pip install chonkie[all]` | All features | All dependencies |

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

# Quick Start

```python
from chonkie import TokenChunker
from tokenizers import Tokenizer

# Initialize tokenizer
tokenizer = Tokenizer.from_pretrained("gpt2")

# Create chunker
chunker = TokenChunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128
)

# Chunk your text
text = """Your long text here..."""
chunks = chunker.chunk(text)

# Access chunks
for chunk in chunks:
    print(f"Chunk: {chunk.text[:50]}...")
    print(f"Tokens: {chunk.token_count}")
```




# Design CHONKosophy

> Did you know that Hippos are surprisingly smart? 

A lot of thought went into this repository and I want to take some space here to fully go over some of the decisions made: the what, why and how? 

1. **Chonkie has it all**: Chonkie comes with the 5 most popularly used chunking approaches, all optimized for your use!
2. **Chonkie is very smart**: By defualt, Chonkie comes with a bunch of very smart hyperparameter choices, that you can just use out of the box. Little tuning and effort required. 
3. **Chonkie is surprisingly lightweight**: Chonkie is meant to be simple and lightweight. Chunking never needed to be complicated, and simple tasks should not take heavy libraries. 
4. **Chonkie is superrrrr fast**: Blazingly fast. Check out our benchmarks!


## Why Chunking is needed? (And may always be needed!)

While you might be aware of models having longer and longer contexts in recent times (as of 2024), models have yet to reach the stage where adding additional context to them comes for free. Additional context, even with the greatest of model architectures comes at a O(n) penalty in speed, to say nothing of the additional memory requirements. And as long as we belive in that attention is all we need, it doesn't seem likely we would be free from this penalty. 

That means, to make models run efficiently (lower speed, memory) it is absoulutely vital that we provide the most accurate information it needs during the retrieval phase. 

Accuracy is one part during retrieval and the other is granularity. You might be able to extract the relevant Article out for model to work with, but if only 1 line is relevant from that passage, you are in effect adding a lot of noise that would hamper and confuse the model in practice. You want and hope to give the model only what it should require ideally (of course, the ideal scenario is rarely ever possible). This finally brings us to granularity and retrieval accuracy. 

Representation models (or embedding models as you may call them) are great at representing large amount of information (sometimes pages of text) in a limited space of just 700-1000 floats, but that doesn't mean it does not suffer from any loss. Most representation is lossy, and if we have many concepts being covered in the same space, it is often that much of it would be lost. However, singluar concepts and explainations breed stronger representation vectors. It then becomes vital again to make sure we don't dilute the representation with noise. 

All this brings me back to chunking. Chunking, done well, can make sure your representation vector (or embedding) is of high-quality to be able to retrieve the best context for your model to generate with. And that in turn, leads to better quality RAG generations. Therefore, I believe chunking is here to stay as long as RAG is here. And hence, it becomes important that we give it little more than a after-thought. 

## Does speed matter while Chunking? (TL;DR: YES!!!)

Human time is limited, and if you have an option that gives you faster chunks, why would you not? 

But speed is not just a bonus; it's central to Chonkie! Whether you are doing RAG on the entirity of Wikipedia or working for large scale organization data that updates regularly, you would need the speed that Chonkie comes with. Stock solutions just don't cut it in these scenarios. 

# Chunkers

## TokenChunker

The `TokenChunker` splits text into chunks based on token count.

```python
from chonkie import TokenChunker
from tokenizers import Tokenizer

chunker = TokenChunker(
    tokenizer=Tokenizer.from_pretrained("gpt2"),
    chunk_size=512,  # Maximum tokens per chunk
    chunk_overlap=128  # Overlap between chunks
)
```

**Key Parameters:**
- `tokenizer`: Any tokenizer implementing the encode/decode interface
- `chunk_size`: Maximum tokens per chunk
- `chunk_overlap`: Number of overlapping tokens between chunks

## WordChunker

The `WordChunker` maintains word boundaries while chunking.

```python
from chonkie import WordChunker

chunker = WordChunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128,
    mode="advanced"  # 'simple' or 'advanced'
)
```

**Key Parameters:**
- `mode`: Chunking mode
  - `simple`: Basic space-based splitting
  - `advanced`: Handles punctuation and special cases

## SentenceChunker

The `SentenceChunker` preserves sentence boundaries.

```python
from chonkie import SentenceChunker

chunker = SentenceChunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128,
    mode="spacy",  # 'simple' or 'spacy'
    min_sentences_per_chunk=1
)
```

**Key Parameters:**
- `mode`: Sentence detection mode
- `min_sentences_per_chunk`: Minimum sentences per chunk

## SemanticChunker

The `SemanticChunker` groups content by semantic similarity.

```python
from chonkie import SemanticChunker

chunker = SemanticChunker(
    tokenizer=tokenizer,
    sentence_transformer_model="all-MiniLM-L6-v2",
    max_chunk_size=512,
    similarity_threshold=0.7
)
```

**Key Parameters:**
- `sentence_transformer_model`: Model for semantic embeddings
- `similarity_threshold`: Threshold for semantic grouping


## SDPMChunker

The `SDPMChunker` groups content via the Semantic Double-Pass Merging method, which groups paragraphs that are semantically similar even if they do not occur consecutively, by making use of a skip-window.

```python
from chonkie import SDPMChunker

chunker = SDPMChunker(
    tokenizer=tokenizer,
    sentence_transformer_model="all-MiniLM-L6-v2",
    max_chunk_size=512,
    similarity_threshold=0.7, 
    skip_window=1
)
```

**Key Parameters:**
- `skip_window`: Size of the skip-window the chunker should pay attention to. Defaults to 1. 

# API Reference

## Chunk Object

```python
@dataclass
class Chunk:
    text: str           # The chunk text
    start_index: int    # Starting position in original text
    end_index: int      # Ending position in original text
    token_count: int    # Number of tokens in chunk
```

## Sentence Chunk Object

```python
@dataclass
class Sentence: 
    text: str
    start_index: int
    end_index: int
    token_count: int

@dataclass
class SentenceChunk(Chunk):
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentences: List[Sentence] 
```

## Semantic Chunk Object

```python
@dataclass
class SemanticSentence(Sentence): 
    text: str
    start_index: int
    end_index: int
    token_count: int
    embedding: Optional[np.ndarray]

@dataclass
class SemanticChunk(SentenceChunk):
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentences: List[SemanticSentence] 
```