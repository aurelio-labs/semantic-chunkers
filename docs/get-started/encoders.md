An encoder turns text into vectors. The chunkers compare those vectors to decide where to split, and that is the only thing they ask of an encoder:

```python
class MyEncoder:
    def __call__(self, docs): ...          # one vector per document
    async def acall(self, docs): ...       # the same, without blocking
```

Anything with those two methods works. There is nothing to inherit from.

## OpenAI

`OpenAIEncoder` ships with the library — no extra to install, and no `openai` package:

```python
import os
from semantic_chunkers import OpenAIEncoder, StatisticalChunker

os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"

encoder = OpenAIEncoder()  # text-embedding-3-small
chunker = StatisticalChunker(encoder=encoder)
```

The key comes from `api_key=` or `OPENAI_API_KEY`. Requests go out in batches of 100 documents, and a rate limit or a server fault is retried, waiting as long as the `Retry-After` header asks for.

### Azure and OpenAI-compatible endpoints

Point `base_url` at any service that speaks the `/embeddings` API — Azure OpenAI, vLLM, Ollama, a gateway of your own. It also reads `OPENAI_BASE_URL`:

```python
encoder = OpenAIEncoder(
    name="text-embedding-3-large",
    base_url="https://my-resource.openai.azure.com/openai/v1",
    dimensions=256,
)
```

## A local model

`LocalEncoder` runs sentence-transformers on your machine, so there is no key and no per-call cost. It is behind an extra because torch and the weights are around 2 GB:

```bash
pip install -qU "semantic-chunkers[local]"
```

```python
from semantic_chunkers import LocalEncoder, StatisticalChunker

chunker = StatisticalChunker(encoder=LocalEncoder("all-MiniLM-L6-v2"))
```

## Anything else

If you already have a function that embeds a list of strings, wrap it. `CallableEncoder` supplies the async path by running your function in a worker thread:

```python
from semantic_chunkers import CallableEncoder, StatisticalChunker

encoder = CallableEncoder(lambda docs: my_model.embed(docs), name="my-model")
chunker = StatisticalChunker(encoder=encoder)
```

Encoders from other libraries, including Semantic Router's, already have `__call__` and `acall`, so they plug straight in:

```python
from semantic_router.encoders import CohereEncoder  # pip install semantic-router

chunker = StatisticalChunker(encoder=CohereEncoder())
```

## When the async path is missing

Not every encoder has one. If yours has no `acall`, `chunker.acall(...)` raises `EncoderError` naming it, rather than failing somewhere inside the chunker:

```python
chunks = await chunker.acall(docs=[...])
# EncoderError: MyEncoder has no usable acall(), so this chunker cannot run
# asynchronously. Give it an `async def acall(self, docs)`, call the chunker
# synchronously, or wrap it: CallableEncoder(encoder) runs the synchronous
# call in a thread.
```
