from collections.abc import Generator
from time import sleep

from httpx import Response
from pydantic import NonNegativeFloat

from yafma.base_client import BaseYandexFoundationModelsClient

from .config import ApiEndpoints
from .models import (
    CompletionAPIResponse,
    CompletionRequest,
    CompletionResponse,
    Operation,
    TokenizeRequest,
    TokenizeResponse,
)


class YandexGptClient(BaseYandexFoundationModelsClient):
    """Client for the YandexGPT API. Supports text generation, tokenization, and "asynchronous" operations.

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/)
    """

    def post_completion(
        self,
        request_data: CompletionRequest,
    ) -> CompletionResponse:
        """Make a POST request to the text generation endpoint.

        Args
        ----
        - `request_data` (`CompletionRequest`): Request data

        Returns
        -------
        - `CompletionResponse`: API response

        Raises
        ------
        - `ValueError`: if `stream` is set to `True`
        - Any exceptions raised by the httpx client during the request or other Yandex-specific errors

        Implements
        ----------
        [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#https-request](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#https-request)
        """
        if request_data.completionOptions.stream:
            msg = "`stream` is set to `True`, use `post_completion_stream` instead"
            raise ValueError(msg)
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TEXT_GENERATION,
            request_data=request_data,
        )
        return self._process_modeled_response(response, CompletionAPIResponse).result

    def post_completion_stream(
        self,
        request_data: CompletionRequest,
    ) -> Generator[CompletionResponse, None, None]:
        """Make a streaming POST request to the text generation endpoint.

        Args
        ----
        - `request_data` (`CompletionRequest`): Request data

        Yields
        ------
        - `CompletionResponse`: API response

        Raises
        ------
        - `ValueError`: if `stream` is set to `False`
        - Any exceptions raised by the httpx client during the request or other Yandex-specific errors

        Implements
        ----------
        [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#https-request](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#https-request)
        """
        if not request_data.completionOptions.stream:
            msg = "`stream` is set to `False`, use `post_completion` instead"
            raise ValueError(msg)
        response: Generator[str, None, None] = self._make_stream_request(
            method="post",
            url=ApiEndpoints.TEXT_GENERATION,
            request_data=request_data,
        )
        for chunk in response:
            yield self._process_modeled_response(chunk, CompletionAPIResponse).result

    def post_completion_async(self, request_data: CompletionRequest) -> Operation:
        """Make an async POST request to the text generation endpoint.

        Args
        ----
        - `request_data` (`CompletionRequest`): Request data

        Returns
        -------
        - `Operation`: API operation

        Raises
        ------
        Any exceptions raised by the httpx client during the request or other Yandex-specific errors.

        Implements
        ----------
        [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGenerationAsync/completion#https-request](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGenerationAsync/completion#https-request)
        """
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TEXT_GENERATION_ASYNC,
            request_data=request_data,
        )
        return self._process_modeled_response(response, Operation)

    def get_operation_status(self, operation_id: str) -> Operation:
        """Get the operation by ID.

        Args
        ----
        - `operation_id` (`str`): Operation ID

        Returns
        -------
        - `Operation`: API operation

        Raises
        ------
        Any exceptions raised by the httpx client during the request or other Yandex-specific errors.

        Implements
        ----------
        [yandex.cloud/en/docs/api-design-guide/concepts/operation](https://yandex.cloud/en/docs/api-design-guide/concepts/operation)
        """
        response: Response = self._make_request(
            method="get",
            url=ApiEndpoints.OPERATIONS.format(operation_id=operation_id),
        )
        return self._process_modeled_response(response, Operation)

    def wait_for_completion(
        self,
        operation_id: str,
        poll_interval: NonNegativeFloat = 1.0,
    ) -> CompletionResponse:
        """Wait for the operation to complete and return the response.

        Args
        ----
        - `operation_id` (`str`): Operation ID
        - `poll_interval` (`NonNegativeFloat`, optional): Polling interval in seconds (defaults to `1.0`)

        Returns
        -------
        - `CompletionResponse`: API response

        Notes
        -----
        - This method is a blocking operation by design.

        Implements
        ----------
        [yandex.cloud/en/docs/api-design-guide/concepts/operation](https://yandex.cloud/en/docs/api-design-guide/concepts/operation)
        """
        while True:
            operation: Operation = self.get_operation_status(operation_id)
            if operation.done and operation.response:
                return operation.response
            elif operation.error:
                msg: str = (
                    f"Operation #{operation_id} failed with error: {operation.error}"
                )
                raise RuntimeError(msg)
            sleep(poll_interval)

    def post_tokenize(self, request_data: TokenizeRequest) -> TokenizeResponse:
        """Make a POST request to the tokenize endpoint.

        Args
        ----
        - `request_data` (`TokenizeRequest`): Request data

        Returns
        -------
        - `TokenizeResponse`: API response

        Implements
        ----------
        [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenize#https-request](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenize#https-request)
        """
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TOKENIZE,
            request_data=request_data,
        )
        return self._process_modeled_response(response, TokenizeResponse)

    def post_tokenize_completion(
        self,
        request_data: CompletionRequest,
    ) -> TokenizeResponse:
        """Make a POST request to the tokenize completion endpoint.

        Args
        ----
        - `request_data` (`CompletionRequest`): Request data

        Returns
        -------
        - `TokenizeResponse`: API response

        Implements
        ----------
        [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenizeCompletion#https-request](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenizeCompletion#https-request)
        """
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TOKENIZE_COMPLETION,
            request_data=request_data,
        )
        return self._process_modeled_response(response, TokenizeResponse)
