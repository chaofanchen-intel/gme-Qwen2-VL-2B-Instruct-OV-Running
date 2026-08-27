class FakeBackend:
    model_name = "fake"
    embedding_dim = 2

    def encode_queries(self, texts, batch_size=None):
        return texts

    def encode_page_images(self, images, batch_size=None):
        return images

    def encode_layout_texts(self, texts, batch_size=None):
        return texts

    def encode_layout_fused(self, texts, images, batch_size=None):
        return texts, images

    def metadata(self):
        return {"value": self.value}


def create(value):
    backend = FakeBackend()
    backend.value = value
    return backend