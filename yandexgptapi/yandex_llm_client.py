from json import loads as json_loads
from time import sleep
from typing import Any, Generator

from httpx import Client, Response
from pydantic import BaseModel

from .config import APIEndpointsV1
from .models import (
    CompletionAPIResponse,
    CompletionRequest,
    CompletionResponse,
    Operation,
    TokenizeRequest,
    TokenizeResponse,
)


class YandexLLMClient:
    """Client for interacting with the Yandex Foundation Models API. This client supports various operations such as text generation, tokenization, and asynchronous operations (as in low priority requests with lowered prices that can take some time to complete).

    Attributes
    ----------
        headers (dict[str, str]): The headers to be used in the API requests. Note: setting the headers updates the values in the client.
        _httpx_client_options (dict): Additional options for the httpx client. Note: hidden attribute; pass these options as arguments to the client constructor.

    Methods
    -------
        __enter__: Context management method to initialize the httpx client.
        __exit__: Context management method to close the httpx client.
        _make_request: Make an API request and returns the response. Note: hidden method.
        _make_stream_request: Make a streaming API request and yields the response. Note: hidden method.
        post_completion: Make a POST request to the text generation endpoint and returns the response.
        post_completion_stream: Make a streaming POST request to the text generation endpoint and yields the response.
        post_completion_async: Make an asynchronous POST request to the text generation endpoint and returns the operation.
        get_operation_status: Get the status of an operation.
        wait_for_completion: Wait for an operation to complete and returns the response.
        post_tokenize: Make a POST request to the tokenize endpoint and returns the response.
        post_tokenize_completion: Make a POST request to the tokenize completion endpoint and returns the response.

    """

    def __init__(
        self,
        folder_id: str | None = None,
        iam_token: str | None = None,
        api_key: str | None = None,
        data_logging_enabled: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the YandexLLMClient with the provided parameters.

        Args:
        ----
            folder_id (str, optional): The ID of the folder in Yandex Cloud. Required when using IAM token.
            iam_token (str, optional): The IAM token for authentication.
            api_key (str, optional): The API key for authentication.
            data_logging_enabled (bool, optional): Whether data logging is enabled. Defaults to False.
            **kwargs: Additional options for the httpx client.

        Raises:
        ------
            ValueError: If neither iam_token nor api_key is provided.
            ValueError: If folder_id is not provided when using iam_token.
            ValueError: If both iam_token and api_key are provided.

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

    def __enter__(self) -> "YandexLLMClient":
        """Context management method to initialize the httpx client.

        Returns
        -------
            YandexLLMClient: The instance of the client.

        """
        self._client = Client(headers=self._headers, **self._httpx_client_options)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """Context management method to close the httpx client."""
        self._client.close()

    @property
    def headers(self) -> dict[str, str]:
        """Getter for the headers attribute.

        Returns
        -------
            dict[str, str]: The headers used in the API requests.

        """
        return self._headers

    @headers.setter
    def headers(self, new_headers: dict[str, str]) -> None:
        """Setter for the headers attribute. Updates the headers in the client.

        Args:
        ----
            new_headers (dict[str, str]): The new headers to be used in the API requests.

        """
        self._headers = new_headers
        self._client.headers.update(new_headers)

    def _make_request(
        self,
        method: str,
        url: str,
        request_data: BaseModel | None = None,
    ) -> Response:
        """Make an API request and returns the response.

        Args:
        ----
            method (str): The HTTP method to use for the request. Corresponds to the method of the httpx client.
            url (str): The URL to send the request to.
            request_data (BaseModel, optional): The data to send in the request. Will be serialized to Python's built-in dict to later be serialized to JSON in httpx. Defaults to None.

        Returns:
        -------
            Response: The response from the API.

        Raises:
        ------
            Whatever exceptions are raised by the httpx client.

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
        """Make a streaming API request and yields the response.

        Args:
        ----
            method (str): The HTTP method to use for the request. Corresponds to the method of the httpx client.
            url (str): The URL to send the request to.
            request_data (BaseModel, optional): The data to send in the request. Will be serialized to Python's built-in dict to later be serialized to JSON in httpx. Defaults to None.

        Yields:
        ------
            str: The response from the API.

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
        """Make a POST request to the text generation endpoint and returns the response.

        Args:
        ----
            request_data (CompletionRequest): The data to send in the request (includes model URI, completion options (such as temperature and max tokens), and messages to generate text from).

        Returns:
        -------
            CompletionResponse: The response from the API (includes generated text, usage, and model version).

        Implements:
        -----------
            https://cloud.yandex.ru/en/docs/yandexgpt/api-ref/v1/TextGeneration/completion

        """
        request_data.completionOptions.stream = False
        response: Response = self._make_request(
            method="post",
            url=APIEndpointsV1.TEXT_GENERATION,
            request_data=request_data,
        )
        parsed_response = CompletionAPIResponse(**response.json())
        return parsed_response.result

    def post_completion_stream(
        self,
        request_data: CompletionRequest,
    ) -> Generator[CompletionResponse, None, None]:
        """Make a streaming POST request to the text generation endpoint and yields the response.

        Args:
        ----
            request_data (CompletionRequest): The data to send in the request (includes model URI, completion options (such as temperature and max tokens), and messages to generate text from). Note: it sets the stream option to True in the request data in any case.

        Yields:
        ------
            CompletionResponse: The response from the API (includes generated text (partially or fully), usage, and model version).

        Implements:
        -----------
            https://cloud.yandex.ru/en/docs/yandexgpt/api-ref/v1/TextGeneration/completion

        """
        request_data.completionOptions.stream = True
        response: Generator[str, None, None] = self._make_stream_request(
            method="post",
            url=APIEndpointsV1.TEXT_GENERATION,
            request_data=request_data,
        )
        for chunk in response:
            parsed_response = CompletionAPIResponse(**json_loads(chunk))
            yield parsed_response.result

    def post_completion_async(self, request_data: CompletionRequest) -> Operation:
        """Make an syncronous "async" POST request to the text generation endpoint and returns the operation. Note: this method is created for low priority requests and can take "some time" to complete, in exchange for "beter quality" and lower prices.

        Args:
        ----
            request_data (CompletionRequest): The data to send in the request (includes model URI, completion options (such as temperature and max tokens), and messages to generate text from).

        Returns:
        -------
            Operation: The operation from the API (includes ID, description, creation and modification times, status, metadata (optional), error (optional), and response (optional)).

        Implements:
        -----------
            https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion

        """
        response: Response = self._make_request(
            method="post",
            url=APIEndpointsV1.TEXT_GENERATION_ASYNC,
            request_data=request_data,
        )
        return Operation(**response.json())

    def get_operation_status(self, operation_id: str) -> Operation:
        """Get the status of an operation.

        Args:
        ----
            operation_id (str): The ID of the operation.

        Returns:
        -------
            Operation: The operation from the API (includes ID, description, creation and modification times, status, metadata (optional), error (optional), and response (optional)).

        Implements:
        -----------
            https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion

        """
        response: Response = self._make_request(
            method="get",
            url=APIEndpointsV1.OPERATIONS.format(operation_id=operation_id),
        )
        return Operation(**response.json())

    def wait_for_completion(
        self,
        operation_id: str,
        poll_interval: float = 1.0,
    ) -> CompletionResponse:
        """Wait for an operation to complete and returns the response. It polls the operation status at regular intervals until the operation is done. Note: this method is a blocking operation by design.

        Args:
        ----
            operation_id (str): The ID of the operation.
            poll_interval (float, optional): The interval in seconds between each poll. Defaults to 1.0.

        Returns:
        -------
            CompletionResponse: The response from the API (includes generated text, usage, and model version).

        Implements:
        -----------
            https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/TextGenerationAsync/completion

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
        """Make a POST request to the tokenize endpoint and returns the response.

        Args:
        ----
            request_data (TokenizeRequest): The data to send in the request (includes model URI and text to tokenize).

        Returns:
        -------
            TokenizeResponse: The response from the API (includes tokens and model version).

        Implements:
        -----------
            https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/Tokenizer/tokenize

        """
        response: Response = self._make_request(
            method="post",
            url=APIEndpointsV1.TOKENIZE,
            request_data=request_data,
        )
        return TokenizeResponse(**response.json())

    def post_tokenize_completion(
        self,
        request_data: CompletionRequest,
    ) -> TokenizeResponse:
        """Make a POST request to the tokenize completion endpoint and returns the response. Note: this method is designed to be a drop-in replacement for the `PostCompletion` method.

        Args:
        ----
            request_data (CompletionRequest): The data to send in the request (includes model URI, completion options (such as temperature and max tokens), and messages to generate text from). Note: it uses only model information and messages from the request data.

        Returns:
        -------
            TokenizeResponse: The response from the API (includes tokens and model version).

        Implements:
        -----------
            https://cloud.yandex.ru/ru/docs/yandexgpt/api-ref/v1/Tokenizer/tokenizeCompletion

        """
        response: Response = self._make_request(
            method="post",
            url=APIEndpointsV1.TOKENIZE_COMPLETION,
            request_data=request_data,
        )
        return TokenizeResponse(**response.json())
