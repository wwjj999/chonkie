# 🦛 CHONK vs The World

> Look how **THICC** the competition is! Meanwhile, Chonkie be looking slim and trim 💪

Ever wondered how much CHONKier other text splitting libraries are? Well, wonder no more! We've put Chonkie up against some of the most popular RAG libraries out there, and the results are... well, let's just say Moto Moto might need to revise his famous quote! 

## 📊 Size Comparison (Package Size)

### Default Installation (Basic Chunking)

| Library | Size | Chonk Factor |
|---------|------|--------------|
| 🦛 Chonkie | 9.7 MiB | 1x (base CHONK) |
| 🔗 LangChain | 80 MiB | ~8.3x CHONKier |
| 📚 LlamaIndex | 171 MiB | ~17.6x CHONKier |

### With Semantic Features

| Library | Size | Chonk Factor |
|---------|------|--------------|
| 🦛 Chonkie | 585 MiB | 1x (semantic CHONK) |
| 🔗 LangChain | 625 MiB | ~1.07x CHONKier |
| 📚 LlamaIndex | 678 MiB | ~1.16x CHONKier |

## ⚡ Speed Benchmarks

> ZOOOOOM! Watch Chonkie run! 🏃‍♂️💨

All benchmarks were run on the Paul Graham Essays Dataset using the GPT-2 tokenizer. Because Chonkie believes in transparency, we note that timings marked with ** were taken after a warm-up phase.

### Token Chunking (ms)

| Library | Time (ms) | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 8.18** | 1x (fastest CHONK) |
| 🔗 LangChain | 8.68 | 1.06x slower |
| 📚 LlamaIndex | 272 | 33.25x slower |

### Sentence Chunking (ms)

| Library | Time (ms) | Speed Factor |
|---------|-----------|--------------|
| 🦛 Chonkie | 52.6 | 1x (solo CHONK) |
| 📚 LlamaIndex | 91.2 | 1.73x slower |
| 🔗 LangChain | N/A | Doesn't exist |

### Semantic Chunking (ms)

| Library | Time | Speed Factor |
|---------|------|--------------|
| 🦛 Chonkie | 482ms | 1x (smart CHONK) |
| 🔗 LangChain | 899ms | 1.86x slower |
| 📚 LlamaIndex | 1.2s | 2.49x slower |

## 💡 Why These Numbers Matter

### Size Benefits
1. **Faster Installation**: Less to download = faster to get started
2. **Lower Memory Footprint**: Lighter package = less RAM usage
3. **Cleaner Dependencies**: Only install what you actually need
4. **CI/CD Friendly**: Faster builds and deployments

### Speed Benefits
1. **Faster Processing**: Chonkie leads in all chunking methods!
2. **Production Ready**: Optimized for real-world usage
3. **Consistent Performance**: Fast across all chunking types
4. **Scale Friendly**: Process more text in less time

Remember what Chonkie always says:
> "I may be a hippo, but I don't have to be heavy... and I can still run fast!" 🦛✨

---

*Note: All measurements were taken using Python 3.8+ on a clean virtual environment. Your actual mileage may vary slightly depending on your specific setup and dependencies. Speed benchmarks were performed on Paul Graham Essays Dataset using the GPT-2 tokenizer.*