Install the library:

```bash
pip install -qU semantic-chunkers
```

That is the whole install. The OpenAI encoder comes with it.

## Pick an encoder and a chunker

The encoder turns sentences into vectors. The chunker uses those vectors to decide where to split. We'll use OpenAI for the encoder and `StatisticalChunker` — the best default for text.

```python
import os
from semantic_chunkers import OpenAIEncoder, StatisticalChunker

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

encoder = OpenAIEncoder()
chunker = StatisticalChunker(encoder=encoder)
```

Prefer a local model, another provider, or a function of your own? See [encoders](encoders). Anything with `__call__(docs)` and `acall(docs)` works, so encoders from other libraries — Semantic Router's included — plug in unchanged.

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

## Sentence splits in other languages, with sentencex

The chunkers start by splitting a document into sentences, with a regex by default. The regex only starts a new sentence before an ASCII capital, so it leaves Russian or Japanese text whole, and French or German sentences that open with `É`, `Ç` or `Ö` stay glued to the one before. It also cuts after any colon or abbreviation followed by a capital: `"Note: This only applies to version 2."` becomes two sentences and `"Dr. Smith arrived"` loses its `"Dr."`. `SentencexSplitter` uses [sentencex](https://github.com/wikimedia/sentencex), Wikimedia's rule-based segmenter with abbreviation lists for many languages, instead:

```bash
pip install -qU "semantic-chunkers[sentencex]"
```

```python
from semantic_chunkers import SentencexSplitter, StatisticalChunker

splitter = SentencexSplitter(language="fr")  # default "en"
chunker = StatisticalChunker(encoder=encoder, splitter=splitter)
chunks = chunker(docs=["Il a quitté Paris. Élise est restée. Ça va mieux."])
```

It works with `StatisticalChunker`, `ConsecutiveChunker` and `CumulativeChunker`; `RegexChunker` uses its own splitter. Its splits carry exact offsets into the document, like the regex's.

**What it costs.** The install is a single compiled package of about 3 MB with no dependencies, with prebuilt wheels for Linux, macOS and Windows on Python 3.11 and later. On Python 3.10 it has no wheel for macOS, nor for Windows in its latest releases, so pip builds it from source there and needs a recent Rust toolchain. On a laptop, after a first call of a few milliseconds that loads it, it split a 1,000-word English document in well under a millisecond, faster than the regex.

**Set the language.** Abbreviations and ordinals are language-specific, so German `"am 3. Oktober"` is split after `"3."` under the default `"en"`. Codes are lower-case Wikipedia-style codes such as `"de"`, `"pt-br"` or `"zh-hans"`. A code sentencex has no rules for, including an upper-case one like `"DE"`, falls back to the closest language it knows and finally to English, without an error.

**Where it falls short.** Its rules lean towards leaving two sentences joined rather than cutting one wrongly:

- It does not split inside a quotation, nor after a list marker such as `(a)` or `(1)` that opens a line, until the next blank line. `"(c) Municipal. This is one. This is two."` comes back as one sentence, so a long quotation or list item can become one very long split.
- It under-splits when unsure: `"Das war usw. Klar."` stays one sentence in German, and a line without closing punctuation, such as a heading followed by a single newline, is joined to the sentence after it. A blank line always ends a sentence.
- It still gets some cases wrong: `"e.g. Paris is big. i.e. It has many people."` gets `"i.e."` split off on its own, a numbered list like `"1. First item. 2. Second item."` comes back with every number as its own sentence, and in Japanese a quotation that ends in `。」` is cut from the `と彼は言った` that follows it.

## Next steps

Try each strategy on your own documents and compare the chunks. The [intro notebook](https://github.com/aurelio-labs/semantic-chunkers/blob/main/docs/00-chunkers-intro.ipynb) runs all of them side by side.
