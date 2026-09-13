Install the library:

```bash
pip install -qU semantic-chunkers
```

Encoders come from Semantic Router, which installs alongside it.

## Pick an encoder and a chunker

The encoder turns sentences into vectors. The chunker uses those vectors to decide where to split. We'll use OpenAI for the encoder and `StatisticalChunker` — the best default for text.

```python
import os
from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import OpenAIEncoder

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

encoder = OpenAIEncoder()
chunker = StatisticalChunker(encoder=encoder)
```

Any Semantic Router encoder works here — `CohereEncoder`, `HuggingFaceEncoder`, `FastEmbedEncoder`, and so on.

## Chunk a document

Pass a list of documents. You get back a list of chunks for each one:

```python
chunks = chunker(docs=["your long document text goes here..."])
```

`chunks[0]` is the list of `Chunk` objects for the first document. Each holds its `splits` (the sentences it contains), a `token_count`, and — for the semantic chunkers — the `triggered_score` that caused the split. To see them at a glance:

```python
chunker.print(chunks[0])
```

## Find a chunk in the document

`content` is the document's own text, whitespace and all, and `start` and `end` are where that text sits in it:

```python
doc = "Alpha one.\n\n  Alpha two. Beta one."
chunks = chunker(docs=[doc])[0]

first = chunks[0]
first.content                                     # the document from first.start to first.end
doc[first.start : first.end] == first.content     # True
"".join(chunk.content for chunk in chunks) == doc # True
```

Because the chunks of a document join back into that document, you can highlight a chunk where it was found, or store an offset instead of a copy of the text. The `splits` stay stripped, so use `content` when you want the layout and `splits` when you want the sentences.

A chunk the library could not place in a document — one you built yourself, or one whose splits are video frames rather than text — has `content`, `start` and `end` set to `None`.

## Go async

Chunking embeds a lot of sentences, so the async path is dramatically faster when you're calling an API. Every chunker has `acall`:

```python
chunks = await chunker.acall(docs=["your long document text goes here..."])
```

The [async notebook](https://github.com/aurelio-labs/semantic-chunkers/blob/main/docs/02-chunkers-async.ipynb) shows it running around 40× faster.

## The strategies

Four chunkers, each a different way of deciding where to split.

```python
from semantic_chunkers import (
    StatisticalChunker,
    ConsecutiveChunker,
    CumulativeChunker,
    RegexChunker,
)

# adaptive threshold per document; the recommended default for text
chunker = StatisticalChunker(encoder=encoder)

# fixed similarity threshold between neighbouring sentences
chunker = ConsecutiveChunker(encoder=encoder, score_threshold=0.45)

# compares each new sentence to the whole chunk so far; stable but expensive
chunker = CumulativeChunker(encoder=encoder, score_threshold=0.45)

# no encoder; splits on delimiters up to a token budget
chunker = RegexChunker(max_chunk_tokens=300)
```

**Statistical** is where to start. It picks its threshold from the document itself, so you rarely need to tune it. Its most useful knobs are `min_split_tokens` (default 100) and `max_split_tokens` (default 300), which bound chunk size.

**Consecutive** is the simplest and cheapest semantic option, and unlike Statistical it can work on more than text. Lower `score_threshold` for bigger chunks, raise it for smaller ones. The default is 0.45.

**Cumulative** is the most stable on noisy text, because each decision considers the whole chunk so far — but that means far more embedding calls. Reach for it when quality matters more than cost.

**Regex** skips embeddings entirely. Use it when you just need size-bounded splits and don't care about semantics, or as a fast first pass.

## Next steps

Try each strategy on your own documents and compare the chunks. The [intro notebook](https://github.com/aurelio-labs/semantic-chunkers/blob/main/docs/00-chunkers-intro.ipynb) runs all of them side by side.
