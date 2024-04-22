from enum import StrEnum


class EmbeddingModelURI(StrEnum):
    """Enum for the URIs of text embedding models.

    Attributes
    ----------
    - `DOC`: Vectorization of large texts (such as documentation articles) (synchronous only)
    - `QUERY`: Vectorization of short texts (such as search queries) (synchronous only)

    Docs
    ----
    [yandex.cloud/en/docs/foundation-models/concepts/embeddings](https://yandex.cloud/en/docs/foundation-models/concepts/embeddings)
    """

    DOC = "emb://{folder_id}/text-search-doc/latest"
    QUERY = "emb://{folder_id}/text-search-query/latest"


class ApiEndpoints(StrEnum):
    """Enum for the endpoints of the Embeddings API.

    Attributes
    ----------
    - `TEXT_EMBEDDING`: Text embedding

    Docs
    ----
    [yandex.cloud/en/docs/foundation-models/concepts/api][https://yandex.cloud/en/docs/foundation-models/concepts/api]
    """

    TEXT_EMBEDDING = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"

