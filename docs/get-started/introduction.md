Semantic Chunkers splits text into chunks based on *meaning*. Instead of cutting every N characters or at every newline, it uses an embedding model to find the places where the topic actually shifts — and splits there.

That matters most for retrieval. A chunk that starts mid-thought and ends mid-thought is a bad search result. A chunk that holds one coherent idea is a good one. Better chunks mean better RAG.

## How it works

The library splits a document into sentences, embeds them, and looks at how similar each sentence is to its neighbours. Where similarity drops, a new chunk begins. You choose how that decision gets made:

- **`StatisticalChunker`** — the most robust option and the one to start with. It sets the similarity threshold dynamically per document, so it adapts to the text instead of relying on a fixed number. Text only.
- **`ConsecutiveChunker`** — the simplest. Groups consecutive sentences while they stay above a fixed similarity threshold. Works on more than just text.
- **`CumulativeChunker`** — accumulates sentences and compares each new one against everything gathered so far. More stable and noise-resistant, but the most expensive in time and, with API encoders, money.
- **`RegexChunker`** — no embeddings at all. Splits on delimiters up to a token budget. Fast and free, for when semantics don't matter.

The semantic chunkers take an encoder from [Semantic Router](../../semantic-router), so you can use OpenAI, Cohere, Hugging Face, FastEmbed, and the rest.

## Start here

The [quickstart](quickstart) chunks a document in a few lines and compares the strategies.

## Resources

- [GitHub repository](https://github.com/aurelio-labs/semantic-chunkers)
