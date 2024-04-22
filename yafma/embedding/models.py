from pydantic import BaseModel, NonNegativeInt

from .config import EmbeddingModelURI


class EmbeddingsRequest(BaseModel):
    """Model representing the embeddings request.

    Attributes
    ----------
    - `modelUri` (`str` or `EmbeddingModelURI`): ID of the model to use
    - `text` (`str`): Text for which to get embeddings

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/embeddings/api-ref/Embeddings/textEmbedding#https-request](https://yandex.cloud/en/docs/foundation-models/embeddings/api-ref/Embeddings/textEmbedding#https-request)
    """

    modelUri: EmbeddingModelURI | str = EmbeddingModelURI.QUERY
    text: str


class EmbeddingsResponse(BaseModel):
    """Model representing the embeddings response.

    Attributes
    ----------
    - `embedding` (`list[float]`): List of embeddings
    - `numTokens` (`NonNegativeInt`): Number of tokens in the input text
    - `modelVersion` (`str`): Version of the model used (a date in `DD.MM.YYYY` format)

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/embeddings/api-ref/Embeddings/textEmbedding#responses](https://yandex.cloud/en/docs/foundation-models/embeddings/api-ref/Embeddings/textEmbedding#responses)
    """

    embedding: list[float]
    numTokens: NonNegativeInt
    modelVersion: str
