from semantic_chunkers.encoders import FastEmbedEncoder


class TestFastEmbedEncoder:
    def test_fastembed_encoder(self):
        encode = FastEmbedEncoder()
        test_docs = ["This is a test", "This is another test"]

        embeddings = encode(test_docs)
        assert isinstance(embeddings, list)
