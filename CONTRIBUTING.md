# 🦛 Contributing to Chonkie

> "I like them big, I like them CONTRIBUTING" ~ Moto Moto, probably

Welcome fellow CHONKer! We're excited that you want to contribute to Chonkie. Whether you're fixing bugs, adding features, or improving documentation, every contribution makes Chonkie a better library for everyone.

## 🎯 Before You Start

1. **Check the issues**: Look for existing issues or open a new one to start a discussion.
2. **Read the docs**: Familiarize yourself with [Chonkie's docs](https://docs.chonkie.ai) and core [concepts](https://docs.chonkie.ai/getting-started/concepts).
3. **Set up your environment**: Follow our development setup guide below.

## 🚀 Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/your-username/chonkie.git
cd chonkie
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# If working on semantic features, also install semantic dependencies
pip install -e ".[dev,semantic]"

# For all features
pip install -e ".[dev,all]"
```

## 🧪 Running Tests

We use pytest for testing. Our tests are configured via `pyproject.toml`. Before submitting a PR, make sure all tests pass:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_token_chunker.py

# Run tests with coverage
pytest --cov=chonkie
```

## 🎨 Code Style

Chonkie uses [ruff](https://github.com/astral-sh/ruff) for code formatting and linting. Our configuration in `pyproject.toml` enforces:

- Code formatting (`F`)
- Import sorting (`I`)
- Documentation style (`D`)
- Docstring coverage (`DOC`)

```bash
# Run ruff
ruff check .

# Run ruff with auto-fix
ruff check --fix .
```

### Documentation Style

We use Google-style docstrings. Example:

```python
def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    """Split text into chunks of specified size.
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
        
    Raises:
        ValueError: If chunk_size <= 0
    """
    pass
```

## 🚦 Pull Request Process

1. **Branch Naming**: Use descriptive branch names:
   - `feature/description` for new features
   - `fix/description` for bug fixes
   - `docs/description` for documentation changes

2. **Commit Messages**: Write clear commit messages:

```markdown
feat: add batch processing to WordChunker

- Implement batch_process method
- Add tests for batch processing
- Update documentation
```

3. **Dependencies**: If adding new dependencies:
   - Core dependencies go in `project.dependencies`
   - Optional features go in `project.optional-dependencies`
   - Development tools go in the `dev` optional dependency group

## 📦 Project Structure

Chonkie's package structure is:
```
src/
├── chonkie/
    ├── chunker/     # Chunking implementations
    ├── embeddings/  # Embedding implementations
    └── refinery/    # Refinement utilities
```

## 🎯 Where to Contribute

### 1. Good First Issues

Look for issues labeled [`good-first-issue`](https://github.com/chonkie-ai/chonkie/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22). These are great starting points for new contributors.

### 2. Documentation

- Improve existing docs
- Add examples
- Fix typos
- Add tutorials

### 3. Code

- Implement new chunking strategies
- Optimize existing chunkers
- Add new tokenizer support
- Improve test coverage

### 4. Performance

- Profile and optimize code
- Add benchmarks
- Improve memory usage
- Enhance batch processing

### 5. New Features

- Add new features to the library
- Add new optional dependencies
- Look for [FEAT] labels in issues, especially by Chonkie Maintainers

## 🦛 Development Dependencies

Current development dependencies are (as of January 1, 2025):

```toml
[project.optional-dependencies]
dev = [
    "pytest>=6.2.0", 
    "datasets>=1.14.0",
    "transformers>=4.0.0",
    "ruff>=0.0.265"
]
```

Additional optional dependencies:

- `model2vec`: For model2vec embeddings
- `st`: For sentence-transformers
- `openai`: For OpenAI embeddings
- `semantic`: For semantic features
- `all`: All optional dependencies

## 🤝 Code Review Process

1. All PRs need at least one review
2. Maintainers will review for:
   - Code quality (via ruff)
   - Test coverage
   - Performance impact
   - Documentation completeness
   - Adherence to principles

## 💡 Getting Help

- **Questions?** Open an issue or ask in Discord
- **Bugs?** Open an issue or report in Discord
- **Chat?** Join our Discord!
- **Email?** Contact [support@chonkie.ai](mailto:support@chonkie.ai)

## 🙏 Thank You

Every contribution helps make Chonkie better! We appreciate your time and effort in helping make Chonkie the CHONKiest it can be!

Remember:
> "A journey of a thousand CHONKs begins with a single commit" ~ Ancient Proverb, probably
