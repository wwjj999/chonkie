<div align='center'>

![Chonkie Logo](/assets/chonkie_logo_br_transparent_bg.png)

# 🦛 Chonkie ✨

[![PyPI version](https://img.shields.io/pypi/v/chonkie.svg)](https://pypi.org/project/chonkie/)
[![License](https://img.shields.io/github/license/bhavnicksm/chonkie.svg)](https://github.com/bhavnicksm/chonkie/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-chonkie.ai-blue.svg)](https://docs.chonkie.ai)
![Package size](https://img.shields.io/badge/size-15MB-blue)
[![codecov](https://codecov.io/gh/chonkie-ai/chonkie/graph/badge.svg?token=V4EWIJWREZ)](https://codecov.io/gh/chonkie-ai/chonkie)
[![Downloads](https://static.pepy.tech/badge/chonkie)](https://pepy.tech/project/chonkie)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/rYYp6DC4cv?style=flat)](https://discord.gg/rYYp6DC4cv)
[![GitHub stars](https://img.shields.io/github/stars/bhavnicksm/chonkie.svg)](https://github.com/bhavnicksm/chonkie/stargazers)

_The no-nonsense RAG chunking library that's lightweight, lightning-fast, and ready to CHONK your texts_

[Installation](#installation) •
[Usage](#usage) •
[Supported Methods](#supported-methods) •
[Benchmarks](#benchmarks-️) •
[Documentation](https://docs.chonkie.ai) •
[Contributing](#contributing)

</div>

Ever found yourself building yet another RAG bot (your 2,342,148th one), only to hit that all-too-familiar wall? You know the one —— where you're stuck choosing between:

- Library X: A behemoth that takes forever to install and probably includes three different kitchen sinks
- Library Y: So bare-bones it might as well be a "Hello World" program
- Writing it yourself? For the 2,342,149th time, _sigh_

And you think to yourself:

> "WHY CAN'T THIS JUST BE SIMPLE?!" </br>
> "Why do I need to choose between bloated and bare-bones?" </br>
> "Why can't I just install, import, and CHONK?!" </br>

Well, look no further than Chonkie! (chonkie boi is a gud boi 🦛💕)

**🚀 Feature-rich**: All the CHONKs you'd ever need </br>
**✨ Easy to use**: Install, Import, CHONK </br>
**⚡ Fast**: CHONK at the speed of light! zooooom </br>
**🌐 Wide support**: Supports all your favorite tokenizer CHONKS </br>
**🪶 Light-weight**: No bloat, just CHONK </br>
**🦛 Cute CHONK mascot**: psst it's a pygmy hippo btw </br>
**❤️ [Moto Moto](#acknowledgements)'s favorite python library** </br>

**Chonkie** is a chunking library that "**just works™**".

# Installation

To install chonkie, simply run:

```bash
pip install chonkie
```

Chonkie follows the rule to have minimal default installs, read the [DOCS](https://docs.chonkie.ai) to know the installation for your required chunker, or simply install `all` if you don't want to think about it (not recommended).

```bash
pip install chonkie[all]
```

# Usage

Here's a basic example to get you started:

```python
# First import the chunker you want from Chonkie 
from chonkie import TokenChunker

# Import your favorite tokenizer library
# Also supports AutoTokenizers, TikToken and AutoTikTokenizer
from tokenizers import Tokenizer 
tokenizer = Tokenizer.from_pretrained("gpt2")

# Initialize the chunker
chunker = TokenChunker(tokenizer)

# Chunk some text
chunks = chunker("Woah! Chonkie, the chunking library is so cool! I love the tiny hippo hehe.")

# Access chunks
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")
```

More example usages given inside the [DOCS](https://docs.chonkie.ai)

# Supported Methods

Chonkie provides several chunkers to help you split your text efficiently for RAG applications. Here's a quick overview of the available chunkers:

- **TokenChunker**: Splits text into fixed-size token chunks.
- **WordChunker**: Splits text into chunks based on words.
- **SentenceChunker**: Splits text into chunks based on sentences.
- **RecursiveChunker**: Splits text hierarchically using customizable rules to create semantically meaningful chunks.
- **SemanticChunker**: Splits text into chunks based on semantic similarity.
- **SDPMChunker**: Splits text using a Semantic Double-Pass Merge approach.
- **LateChunker (experimental)**: Embeds text and then splits it to have better chunk embeddings.

More on these methods and the approaches taken inside the [DOCS](https://docs.chonkie.ai)

# Benchmarks 🏃‍♂️

> "I may be smol hippo, but I pack a punch!" 🦛

Here's a quick peek at how Chonkie performs:

**Size**📦

- **Default Install:** 15MB (vs 80-171MB for alternatives)
- **With Semantic:** Still lighter than the competition!

**Speed**⚡

- **Token Chunking:** 33x faster than the slowest alternative
- **Sentence Chunking:** Almost 2x faster than competitors
- **Semantic Chunking:** Up to 2.5x faster than others

Check out our detailed [benchmarks](https://docs.chonkie.ai/benchmarks) to see how Chonkie races past the competition! 🏃‍♂️💨

# Contributing

Want to help make Chonkie even better? Check out our [CONTRIBUTING.md](CONTRIBUTING.md) guide! Whether you're fixing bugs, adding features, or improving docs, every contribution helps make Chonkie a better CHONK for everyone.

Remember: No contribution is too small for this tiny hippo! 🦛

# Acknowledgements

Chonkie would like to CHONK its way through a special thanks to all the users and contributors who have helped make this library what it is today! Your feedback, issue reports, and improvements have helped make Chonkie the CHONKIEST it can be.

And of course, special thanks to [Moto Moto](https://www.youtube.com/watch?v=I0zZC4wtqDQ&t=5s) for endorsing Chonkie with his famous quote:
> "I like them big, I like them chonkie."
>                                         ~ Moto Moto


# Citation

If you use Chonkie in your research, please cite it as follows:

```bibtex
@misc{chonkie2024,
  author = {Minhas, Bhavnick AND Nigam, Shreyash},
  title = {Chonkie: A Fast Feature-full Chunking Library for RAG Bots},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/bhavnick/chonkie}},
}
```
