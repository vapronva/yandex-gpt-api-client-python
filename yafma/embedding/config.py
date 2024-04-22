from enum import Enum, StrEnum

from pydantic import BaseModel, PositiveInt


class EmbeddingModelURI(StrEnum):
    """Enum for the URIs of the text embedding models.

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


class ModelLimits(BaseModel):
    """Model for the limits of the embedding models.

    Attributes
    ----------
    - `MAX_TOKENS_INPUT` (`PositiveInt`): Maximum tokens for the input
    """

    MAX_TOKENS_INPUT: PositiveInt = 2000


class GenerativeModelLimits(Enum):
    """Enum for the limits of the embedding models.

    Attributes
    ----------
    - `DOC`: Vectorization of large texts (2000 tokens for input)
    - `QUERY`: Vectorization of short texts (2000 tokens for input)

    Docs
    ----
    [yandex.cloud/en/docs/foundation-models/concepts/limits](https://yandex.cloud/en/docs/foundation-models/concepts/limits)
    """

    DOC = ModelLimits(MAX_TOKENS_INPUT=2000)
    QUERY = ModelLimits(MAX_TOKENS_INPUT=2000)


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
