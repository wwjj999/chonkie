# 🦛 chonkie docs

> ugh, do i _need_ to explain how to use chonkie? man, that's a bummer... to be honest, chonkie is very easy, with little documentation necessary, but just in case, i'll include some here. 

# table of contents

- [🦛 chonkie docs](#-chonkie-docs)
- [table of contents](#table-of-contents)
- [installation](#installation)
  - [basic installation](#basic-installation)
  - [installation options](#installation-options)
  - [dependency table](#dependency-table)
- [quick start](#quick-start)
- [design chonkosophy](#design-chonkosophy)
  - [core chonk principles](#core-chonk-principles)
    - [1. 🎯 small but precise](#1--small-but-precise)
    - [2. 🚀 surprisingly quick](#2--surprisingly-quick)
    - [3. 🪶 tiny but complete](#3--tiny-but-complete)
    - [4. 🧠 the clever chonk](#4--the-clever-chonk)
    - [5. 🌱 growing with purpose](#5--growing-with-purpose)
  - [why chunking is needed? (and may always be needed!)](#why-chunking-is-needed-and-may-always-be-needed)
  - [does speed matter while chunking? (tl;dr: yes!!!)](#does-speed-matter-while-chunking-tldr-yes)
  - [but but but... how? how is chonkie so fast?](#but-but-but-how-how-is-chonkie-so-fast)
- [chunkers](#chunkers)
  - [tokenchunker](#tokenchunker)
  - [wordchunker](#wordchunker)
  - [sentencechunker](#sentencechunker)
  - [semanticchunker](#semanticchunker)
  - [sdpmchunker](#sdpmchunker)
- [api reference](#api-reference)
  - [chunk object](#chunk-object)
  - [sentence chunk object](#sentence-chunk-object)
  - [semantic chunk object](#semantic-chunk-object)
  

# installation

## basic installation
```bash
pip install chonkie
```

## installation options
chonkie uses optional dependencies to keep the base installation lightweight. choose the installation that best fits your needs:

| installation command | use case | dependencies added |
|---------------------|----------|-------------------|
| `pip install chonkie` | basic token and word chunking | tokenizers |
| `pip install chonkie[sentence]` | sentence-based chunking | + spacy |
| `pip install chonkie[semantic]` | semantic chunking | + sentence-transformers, numpy |
| `pip install chonkie[all]` | all features | all dependencies |

## dependency table

as per the details mentioned in the [design](#design-chonkosophy) section, chonkie is lightweight because it keeps most of the dependencies for each chunker seperate, making it more of an aggregate of multiple repositories and python packages. the optional dependencies feature in python really helps with this. 

| chunker  | default | 'sentence' | 'semantic' | 'all' |
|----------|----------|----------|----------|----------|
| tokenchunker        |✅|✅|✅|✅|
| wordchunker         |✅|✅|✅|✅|
| sentencechunker     |⚠️|✅|⚠️|✅|
| semanticchunker     |❌|❌|⚠️/✅|✅|
| spdmchunker         |❌|❌|⚠️/✅|✅|

note: in the above table `⚠️/✅` meant that some features would be disabled but the chunker would work nonetheless. 

what you could infer from the table is that, while it might be of inconvinience in the short-run to have it split like that, you can do surprisingly a lot with just the defualt dependencies (which btw are super light). furthermore, even our max dependencies option `all` is lightweight in comparison to some of the other libraries that one might use for such tasks. 

# quick start

```python
from chonkie import tokenchunker
from tokenizers import tokenizer

# initialize tokenizer
tokenizer = tokenizer.from_pretrained("gpt2")

# create chunker
chunker = tokenchunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128
)

# chunk your text
text = """your long text here..."""
chunks = chunker.chunk(text)

# access chunks
for chunk in chunks:
    print(f"chunk: {chunk.text[:50]}...")
    print(f"tokens: {chunk.token_count}")
```

# design chonkosophy

> did you know that pygmy hippos are only 1/4 the size of regular hippos, but they're just as mighty? that's the chonkie spirit - tiny but powerful! 🦛

listen up chonkers! just like our adorable pygmy hippo mascot, chonkie proves that the best things come in small packages. let's dive into why this tiny chonkster is built the way it is!

## core chonk principles

### 1. 🎯 small but precise

like how pygmy hippos take perfect little bites of their favorite fruits, chonkie knows exactly how to size your chunks:
- **compact & efficient**: just like our tiny mascot, every chunk is exactly the size it needs to be
- **smart defaults**: we've done the research so you don't have to! our default parameters are battle-tested
- **flexible sizing**: because sometimes you need a smaller bite!

### 2. 🚀 surprisingly quick

fun fact: pygmy hippos might be small, but they can zoom through the forest at impressive speeds! similarly, chonkie is:
- **lightning fast**: small size doesn't mean slow performance
- **optimized paths**: like our mascot's forest shortcuts, we take the most efficient route (we use cacheing extensively btw!)
- **minimal overhead**: no wasted energy, just pure chonk power

### 3. 🪶 tiny but complete

just as pygmy hippos pack all hippo features into a compact frame, chonkie is:
- **minimum footprint**: base installation smaller than a pygmy hippo footprint
- **modular growth**: add features as you need them, like a growing hippo
- **zero bloat**: every feature has a purpose, just like every trait of our tiny friend
- **smart imports**: load only what you need, when you need it

### 4. 🧠 the clever chonk

why chunking still matters (from a tiny hippo's perspective):

1. **right-sized processing**
   - even tiny chunks can carry big meaning
   - smart chunking = efficient processing
   - our pygmy hippo philosophy: "just enough, never too much"

2. **the goldilocks zone**
   - too small: like a hippo bite that's too tiny
   - too large: like trying to swallow a whole watermelon
   - just right: the chonkie way™️ (pygmy-approved!)

3. **semantic sense**
   - each chunk is carefully crafted
   - like our mascot's careful step through the forest
   - small, meaningful units that work together

### 5. 🌱 growing with purpose

like how pygmy hippos stay small but mighty, chonkie grows sensibly:
```
smart chunks → better embeddings → precise retrieval → quality generation
```

even as models grow bigger, you'll appreciate our tiny-but-mighty approach:
- focused context (like a pygmy hippo's keen senses)
- efficient processing (like our mascot's energy-saving size)
- clean, purposeful design (like nature's perfect mini-hippo)

## why chunking is needed? (and may always be needed!)

while you might be aware of models having longer and longer contexts in recent times (as of 2024), models have yet to reach the stage where adding additional context to them comes for free. additional context, even with the greatest of model architectures comes at a o(n) penalty in speed, to say nothing of the additional memory requirements. and as long as we belive in that attention is all we need, it doesn't seem likely we would be free from this penalty. 

that means, to make models run efficiently (lower speed, memory) it is absoulutely vital that we provide the most accurate information it needs during the retrieval phase. 

accuracy is one part during retrieval and the other is granularity. you might be able to extract the relevant article out for model to work with, but if only 1 line is relevant from that passage, you are in effect adding a lot of noise that would hamper and confuse the model in practice. you want and hope to give the model only what it should require ideally (of course, the ideal scenario is rarely ever possible). this finally brings us to granularity and retrieval accuracy. 

representation models (or embedding models as you may call them) are great at representing large amount of information (sometimes pages of text) in a limited space of just 700-1000 floats, but that doesn't mean it does not suffer from any loss. most representation is lossy, and if we have many concepts being covered in the same space, it is often that much of it would be lost. however, singluar concepts and explainations breed stronger representation vectors. it then becomes vital again to make sure we don't dilute the representation with noise. 

all this brings me back to chunking. chunking, done well, can make sure your representation vector (or embedding) is of high-quality to be able to retrieve the best context for your model to generate with. and that in turn, leads to better quality rag generations. therefore, i believe chunking is here to stay as long as rag is here. and hence, it becomes important that we give it little more than a after-thought. 

## does speed matter while chunking? (tl;dr: yes!!!)

human time is limited, and if you have an option that gives you faster chunks, why would you not? 

but speed is not just a bonus; it's central to chonkie! whether you are doing rag on the entirity of wikipedia or working for large scale organization data that updates regularly, you would need the speed that chonkie comes with. stock solutions just don't cut it in these scenarios. 

## but but but... how? how is chonkie so fast?

we used a lot of optimizations when building each and every chunker inside chonkie, making sure it's as optimized as possible. 

1. **using tiktoken (as a default):** tiktoken is around 3-6x faster than it's counterparts; and it is blazing fast when used with multiple threads. we see the available threads on the cpu at the moment, and use about ~70-80% of them (so as to not hog all resources), which inturn let's us tokenize fast.
2. **pre-compute and cache:** we never tokenize or embed on the fly! as long as something can be pre-computed and cached we do that, store it and re-use it wherever possible. ram is cheap but time is priceless. (of course, we also provide options to turn off the pre-computation and make it memory efficient if need be)
3. **running mean pooling:** most semantic chunkers re-embed the chunks every time they get updated, but we don't do that. we pre-compute the embeddings for the sentences, and use mathematical trickery (which is theoretically found) to instead have a running mean pooling of tokens -- which allows us to save the cost from the embedding models. 

# chunkers

## tokenchunker

the `tokenchunker` splits text into chunks based on token count.

```python
from chonkie import tokenchunker
from tokenizers import tokenizer

chunker = tokenchunker(
    tokenizer=tokenizer.from_pretrained("gpt2"),
    chunk_size=512,  # maximum tokens per chunk
    chunk_overlap=128  # overlap between chunks
)
```

**key parameters:**
- `tokenizer`: any tokenizer implementing the encode/decode interface
- `chunk_size`: maximum tokens per chunk
- `chunk_overlap`: number of overlapping tokens between chunks

## wordchunker

the `wordchunker` maintains word boundaries while chunking.

```python
from chonkie import wordchunker

chunker = wordchunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128,
    mode="advanced"  # 'simple' or 'advanced'
)
```

**key parameters:**
- `mode`: chunking mode
  - `simple`: basic space-based splitting
  - `advanced`: handles punctuation and special cases

## sentencechunker

the `sentencechunker` preserves sentence boundaries.

```python
from chonkie import sentencechunker

chunker = sentencechunker(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=128,
    mode="spacy",  # 'simple' or 'spacy'
    min_sentences_per_chunk=1
)
```

**key parameters:**
- `mode`: sentence detection mode
- `min_sentences_per_chunk`: minimum sentences per chunk

## semanticchunker

the `semanticchunker` groups content by semantic similarity.

```python
from chonkie import semanticchunker

chunker = semanticchunker(
    tokenizer=tokenizer,
    sentence_transformer_model="all-minilm-l6-v2",
    max_chunk_size=512,
    similarity_threshold=0.7
)
```

**key parameters:**
- `sentence_transformer_model`: model for semantic embeddings
- `similarity_threshold`: threshold for semantic grouping


## sdpmchunker

the `sdpmchunker` groups content via the semantic double-pass merging method, which groups paragraphs that are semantically similar even if they do not occur consecutively, by making use of a skip-window.

```python
from chonkie import sdpmchunker

chunker = sdpmchunker(
    tokenizer=tokenizer,
    sentence_transformer_model="all-minilm-l6-v2",
    max_chunk_size=512,
    similarity_threshold=0.7, 
    skip_window=1
)
```

**key parameters:**
- `skip_window`: size of the skip-window the chunker should pay attention to. defaults to 1. 

# api reference

## chunk object

```python
@dataclass
class chunk:
    text: str           # the chunk text
    start_index: int    # starting position in original text
    end_index: int      # ending position in original text
    token_count: int    # number of tokens in chunk
```

## sentence chunk object

```python
@dataclass
class sentence: 
    text: str
    start_index: int
    end_index: int
    token_count: int

@dataclass
class sentencechunk(chunk):
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentences: list[sentence] 
```

## semantic chunk object

```python
@dataclass
class semanticsentence(sentence): 
    text: str
    start_index: int
    end_index: int
    token_count: int
    embedding: optional[np.ndarray]

@dataclass
class semanticchunk(sentencechunk):
    text: str
    start_index: int
    end_index: int
    token_count: int
    sentences: list[semanticsentence] 
```