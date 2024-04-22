from json import loads as json_loads
from time import sleep
from typing import Any, Generator

from httpx import Client, Response
from pydantic import BaseModel

from .config import ApiEndpoints
from .models import (
    CompletionAPIResponse,
    CompletionRequest,
    CompletionResponse,
    Operation,
    TokenizeRequest,
    TokenizeResponse,
)


class YandexGptClient:
    """Client for Yandex Foundation Models API. Supports text generation, tokenization, and asynchronous operations.

    Attributes
    ----------
        - `headers` (`dict[str, str]`): API request headers (updated on setting)
        - `_httpx_client_options` (`dict`): Extra options for httpx client. Hidden attribute; pass as constructor arguments.

    Methods
    -------
        - `__enter__`: Initializes httpx client (for context management).
        - `__exit__`: Closes httpx client (for context management).
        - `_make_request`: Execute API request and returns response. Hidden method.
        - `_make_stream_request`: Executes streaming API request and yields response. Hidden method.
        - `post_completion`: Executes POST request to text generation endpoint and returns response.
        - `post_completion_stream`: Executes streaming POST request to text generation endpoint and yields response.
        - `post_completion_async`: Executes async POST request to text generation endpoint and returns operation.
        - `get_operation_status`: Retrieves operation status.
        - `wait_for_completion`: Waits for operation completion and returns response.
        - `post_tokenize`: Executes POST request to tokenize endpoint and returns response.
        - `post_tokenize_completion`: Executes POST request to tokenize completion endpoint and returns response.
    """

    def __init__(
        self,
        folder_id: str | None = None,
        iam_token: str | None = None,
        api_key: str | None = None,
        data_logging_enabled: bool = False,
        **kwargs,
    ) -> None:
        """Initialize `YandexGPTClient`.

        Args
        ----
            - `folder_id` (`str`, optional): Yandex Cloud folder ID. Required with IAM token.
            - `iam_token` (`str`, optional): IAM token for authentication.
            - `api_key` (`str`, optional): API key for authentication.
            - `data_logging_enabled` (`bool`, optional): Enables data logging (on the Yandex's side). Defaults to `False`.
            - `**kwargs`: Additional httpx client options.

        Raises
        ------
            - `ValueError`: If neither `iam_token` nor `api_key` is provided.
            - `ValueError`: If `folder_id` is not provided when using `iam_token`.
            - `ValueError`: If both `iam_token` and `api_key` are provided.
        """
        if not iam_token and not api_key:
            msg = "Either iam_token or api_key must be provided"
            raise ValueError(msg)
        if not folder_id and iam_token:
            msg = "folder_id is required when using iam_token"
            raise ValueError(msg)
        if iam_token and api_key:
            msg = "Only one of iam_token or api_key must be provided"
            raise ValueError(msg)
        self._headers: dict[str, str] = {
            "x-folder-id": f"{folder_id}",
            "x-data-logging-enabled": "true" if data_logging_enabled else "false",
            "Authorization": f"Api-Key {api_key}" if api_key else f"Bearer {iam_token}",
        }
        if api_key and not folder_id:
            self._headers.pop("x-folder-id")
        self._httpx_client_options = kwargs or {}

    def __enter__(self) -> "YandexGptClient":
        """Initialize httpx client (for context management).

        Returns
        -------
            `YandexGPTClient`: The client instance.
        """
        self._client = Client(headers=self._headers, **self._httpx_client_options)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Close the httpx client (for context management)."""
        self._client.close()

    @property
    def headers(self) -> dict[str, str]:
        """Return the headers used in API requests.

        Returns
        -------
            `dict[str, str]`: The headers.
        """
        return self._headers

    @headers.setter
    def headers(self, new_headers: dict[str, str]) -> None:
        """Update the headers in the client.

        Args
        ----
            - `new_headers` (`dict[str, str]`): The new headers.
        """
        self._headers = new_headers
        self._client.headers.update(new_headers)

    def _make_request(
        self,
        method: str,
        url: str,
        request_data: BaseModel | None = None,
    ) -> Response:
        """Make an API request and return the response.

        Args
        ----
            - `method` (`str`): HTTP method for the request.
            - `url` (`str`): Request URL.
            - `request_data` (`BaseModel`, optional): Data to send in the request.

        Returns
        -------
            `Response`: API response.

        Raises
        ------
            Any exceptions raised by the httpx client itself.
        """
        request_args: dict[str, Any] = {"url": url}
        if request_data:
            request_args["json"] = request_data.model_dump(mode="python")
        response: Response = getattr(self._client, method)(**request_args)
        response.raise_for_status()
        return response

    def _make_stream_request(
        self,
        method: str,
        url: str,
        request_data: BaseModel | None = None,
    ) -> Generator[str, None, None]:
        """Make a streaming API request and yield the response.

        Args
        ----
            - `method` (`str`): HTTP method for the request.
            - `url` (`str`): Request URL.
            - `request_data` (`BaseModel`, optional): Data to send in the request.

        Yields
        ------
            `str`: API response.
        """
        request_args: dict[str, Any] = {"url": url}
        if request_data:
            request_args["json"] = request_data.model_dump(mode="python")
        with self._client.stream(method, **request_args) as response:
            yield from response.iter_text()

    def post_completion(
        self,
        request_data: CompletionRequest,
    ) -> CompletionResponse:
        """Make a POST request to the text generation endpoint.

        Args
        ----
            - `request_data` (`CompletionRequest`): Request data.

        Returns
        -------
            `CompletionResponse`: API response.

        Implements
        -----------
            [cloud.yandex.ru/en/docs/yandexgpt/api-ref/v1/TextGeneration/completion](https://cloud.yandex.ru/en/docs/yandexgpt/api-ref/v1/TextGeneration/completion)
        """
        request_data.completionOptions.stream = False
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TEXT_GENERATION,
            request_data=request_data,
        )
        parsed_response = CompletionAPIResponse(**response.json())
        return parsed_response.result

    def post_completion_stream(
        self,
        request_data: CompletionRequest,
    ) -> Generator[CompletionResponse, None, None]:
        """Make a streaming POST request to the text generation endpoint.

        Args
        ----
            - `request_data` (`CompletionRequest`): Request data.

        Yields
        ------
            `CompletionResponse`: API response.

        Implements
        -----------
            [cloud.yandex.ru/en/docs/yandexgpt/api-ref/v1/TextGeneration/completion](https://cloud.yandex.ru/en/docs/yandexgpt/api-ref/v1/TextGeneration/completion)
        """
        request_data.completionOptions.stream = True
        response: Generator[str, None, None] = self._make_stream_request(
            method="post",
            url=ApiEndpoints.TEXT_GENERATION,
            request_data=request_data,
        )
        for chunk in response:
            parsed_response = CompletionAPIResponse(**json_loads(chunk))
            yield parsed_response.result

    def post_completion_async(self, request_data: CompletionRequest) -> Operation:
        """Make an async POST request to the text generation endpoint.
        Note: this method is designed for low priority requests and can take "some time" to complete, in exchange for "beter quality" and lower prices.

        Args
        ----
            - `request_data` (`CompletionRequest`): Request data.

        Returns
        -------
            `Operation`: API operation.

        Implements
        -----------
            [cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion](https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion)
        """
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TEXT_GENERATION_ASYNC,
            request_data=request_data,
        )
        return Operation(**response.json())

    def get_operation_status(self, operation_id: str) -> Operation:
        """Get the status of an operation.

        Args
        ----
            - `operation_id` (`str`): Operation ID.

        Returns
        -------
            `Operation`: API operation.

        Implements
        -----------
            [cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion](https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion)
        """
        response: Response = self._make_request(
            method="get",
            url=ApiEndpoints.OPERATIONS.format(operation_id=operation_id),
        )
        return Operation(**response.json())

    def wait_for_completion(
        self,
        operation_id: str,
        poll_interval: float = 1.0,
    ) -> CompletionResponse:
        """Wait for an operation to complete and return the response.
        Note: this method is a blocking operation by design.

        Args
        ----
            `operation_id` (`str`): Operation ID.
            `poll_interval` (`float`, optional): Polling interval in seconds. Defaults to `1.0`.

        Returns
        -------
            `CompletionResponse`: API response.

        Implements
        -----------
            [cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion](https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion)
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
            - `request_data` (`TokenizeRequest`): Request data.

        Returns
        -------
            `TokenizeResponse`: API response.

        Implements
        -----------
            [cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/Tokenizer/tokenize](https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/Tokenizer/tokenize)
        """
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TOKENIZE,
            request_data=request_data,
        )
        return TokenizeResponse(**response.json())

    def post_tokenize_completion(
        self,
        request_data: CompletionRequest,
    ) -> TokenizeResponse:
        """Make a POST request to the tokenize completion endpoint.

        Args
        ----
            - `request_data` (`CompletionRequest`): Request data.

        Returns
        -------
            `TokenizeResponse`: API response.

        Implements
        -----------
            [cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/Tokenizer/tokenizeCompletion](https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/Tokenizer/tokenizeCompletion)
        """
        response: Response = self._make_request(
            method="post",
            url=ApiEndpoints.TOKENIZE_COMPLETION,
            request_data=request_data,
        )
        return TokenizeResponse(**response.json())
