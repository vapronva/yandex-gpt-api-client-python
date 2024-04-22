from json import loads as json_loads
from time import sleep
from typing import Any, Generator

from httpx import Client, Response
from pydantic import BaseModel, NonNegativeFloat

from yafma.errors import BaseYandexFoundationModelsApiError, QuotaExceededError

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
    """Client for the Yandex Foundation Models API. Supports text generation, tokenization, and "asynchronous" operations.

    Attributes
    ----------
    - `headers` (`dict[str, str]`): Additional headers for the API requests (updated on setting)
    - `_httpx_client_options` (`dict[str, Any]`): Extra options for httpx client (hidden attribute; pass as constructor arguments when initializing the client)

    Methods
    -------
    - `__enter__`: Initialize httpx client (for context management)
    - `__exit__`: Close httpx client (for context management)
    - `_process_raw_response`: Process the API response and raise an error if needed (hidden method)
    - `_process_modeled_response`: Process a response and return an instance of the expected type (hidden method)
    - `_make_request`: Execute an API request and return the response (hidden method)
    - `_make_stream_request`: Execute a streaming API request and yield the response (hidden method)
    - `post_completion`: Execute a POST request to the text generation endpoint and return the response
    - `post_completion_stream`: Execute a streaming POST request to the text generation endpoint and yield the response
    - `post_completion_async`: Execute an async POST request to the text generation endpoint and return the operation
    - `get_operation_status`: Retrieve the operation
    - `wait_for_completion`: Wait for the operation completion and return the response
    - `post_tokenize`: Execute a POST request to the tokenization endpoint and return the response
    - `post_tokenize_completion`: Execute a POST request to the tokenization completion endpoint and return the response

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/)
    """

    def __init__(
        self,
        folder_id: str | None = None,
        iam_token: str | None = None,
        api_key: str | None = None,
        data_logging_enabled: bool = False,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initialize `YandexGptClient`.

        Args
        ----
        - `folder_id` (`str`, optional): Yandex Cloud folder ID (required with IAM token)
        - `iam_token` (`str`, optional): IAM token for authentication
        - `api_key` (`str`, optional): API key for authentication
        - `data_logging_enabled` (`bool`, optional): Enables data logging on the Yandex's side (by default Yandex always logs the data)
        - `**kwargs` (`dict[str, Any]`, optional): Extra options for the httpx client

        Raises
        ------
        - `ValueError`:
            - if neither `iam_token` nor `api_key` is provided
            - if `folder_id` is not provided when using `iam_token`
            - if both `iam_token` and `api_key` are provided
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
            _ = self._headers.pop("x-folder-id")
        self._httpx_client_options: dict[str, Any] = kwargs or {}
        super().__init__()

    def __enter__(self) -> "YandexGptClient":
        """Initialize httpx client in the context manager.

        Returns
        -------
        - `YandexGptClient`: The client instance
        """
        self._client = Client(headers=self._headers, **self._httpx_client_options)  # type: ignore[reportAny]
        return self

    def __exit__(self, *args: object, **kwargs: dict[str, Any]) -> None:
        """Close the httpx client in the context manager."""
        self._client.close()

    @property
    def headers(self) -> dict[str, str]:
        """Return the headers used in API requests.

        Returns
        -------
        - `dict[str, str]`: The headers
        """
        return self._headers

    @headers.setter
    def headers(self, new_headers: dict[str, str]) -> None:
        """Update the headers used in API requests.

        Args
        ----
        - `new_headers` (`dict[str, str]`): The new headers to set

        Notes
        -----
        - This method is used to update the headers in the client instance, rather than re-assiging the `headers` attribute.
        """
        self._headers.update(new_headers)

    @staticmethod
    def _process_raw_response(response: Response) -> None:
        """Process the API response and raise an error if needed.

        Args
        ----
        - `response` (`Response`): API response

        Raises
        ------
        - `QuotaExceededError`: If the quota is exceeded
        - `BaseYandexFoundationModelsApiError`: If the response status code is greater than or equal to 400
        """
        match response.status_code:
            case 429:
                raise QuotaExceededError()
            case _:
                if response.status_code >= 400:
                    raise BaseYandexFoundationModelsApiError(
                        grpc_code=None,
                        http_code=response.status_code,
                        message=response.text,
                        details=[],
                        solution=None,
                    )

    @staticmethod
    def _process_modeled_response[T: BaseModel](
        response: Response | str,
        expected_type: type[T],
    ) -> T:
        """Process a response and return an instance of the expected type.

        Args
        ----
        - `response` (Response): The response to process.
        - `expected_type` (`Type[T]`): The expected type of the response.

        Returns
        -------
        - An instance of the expected type `T`.

        Raises
        ------
        - `ValueError`: if the response is not a dict or cannot be parsed into the expected type
        """
        if isinstance(response, str):
            parsed_response: Any = json_loads(response)
        else:
            parsed_response: Any = response.json()
        if isinstance(parsed_response, dict):
            return expected_type(**parsed_response)
        msg = f"Invalid response received: {response.text if isinstance(response, Response) else response}"
        raise ValueError(msg)

    def _make_request(
        self,
        method: str,
        url: str,
        request_data: BaseModel | None = None,
    ) -> Response:
        """Make an API request and return the response.

        Args
        ----
        - `method` (`str`): HTTP method for the request
        - `url` (`str`): Request URL
        - `request_data` (`BaseModel`, optional): Data to send in the request (request body)

        Returns
        -------
        - `Response`: API response

        Raises
        ------
        Any exceptions raised by the httpx client or other Yandex-specific errors.
        """
        request_args: dict[str, Any] = {
            "url": url,
            "headers": self._headers,
        }
        if request_data:
            request_args["json"] = request_data.model_dump(mode="python")
        response: Response = getattr(self._client, method)(**request_args)
        self._process_raw_response(response)
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
        - `method` (`str`): HTTP method for the request
        - `url` (`str`): Request URL
        - `request_data` (`BaseModel`, optional): Data to send in the request (request body)

        Yields
        ------
        - `str`: API response

        Raises
        ------
        Any exceptions raised by the httpx client or other Yandex-specific errors.
        """
        request_args: dict[str, Any] = {
            "url": url,
            "headers": self._headers,
        }
        if request_data:
            request_args["json"] = request_data.model_dump(mode="python")
        with self._client.stream(method=method, **request_args) as response:  # type: ignore[reportAny]
            self._process_raw_response(response)
            yield from response.iter_text()

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
