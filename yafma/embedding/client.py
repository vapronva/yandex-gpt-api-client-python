from typing import override

from httpx import Response

from yafma.base_client import BaseYandexFoundationModelsClient

from .config import ApiEndpoints
from .models import (
    EmbeddingsRequest,
    EmbeddingsResponse,
)


class EmbeddingsClient(BaseYandexFoundationModelsClient):
    """Client for the Yandex's Embeddings API.

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/embeddings/api-ref/](https://yandex.cloud/en/docs/foundation-models/embeddings/api-ref/)
    """

    @override
    def __enter__(self) -> "EmbeddingsClient":
        """Initialize the context manager (and httpx client).

        Returns
        -------
        - `EmbeddingsClient`: The client instance
        """
        _ = super().__enter__()
        return self

    def post_embedding(self, request_data: EmbeddingsRequest) -> EmbeddingsResponse:
        """Make a POST request to the text embedding endpoint.

        Args
        ----
        - `request_data` (`EmbeddingsRequest`): Request data

        Returns
        -------
        - `EmbeddingsResponse`: API response

        Implements
        ----------
        [yandex.cloud/en/docs/foundation-models/embeddings/api-ref/Embeddings/textEmbedding](https://yandex.cloud/en/docs/foundation-models/embeddings/api-ref/Embeddings/textEmbedding)
        """
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TEXT_EMBEDDING,
            request_data=request_data,
        )
        return self._process_modeled_response(response, EmbeddingsResponse)
