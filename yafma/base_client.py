from collections.abc import Generator
from json import loads as json_loads
from typing import Any

from httpx import Client, Response
from pydantic import BaseModel

from yafma.errors import BaseYandexFoundationModelsApiError, QuotaExceededError


class BaseYandexFoundationModelsClient:
    """Base client for the Yandex Foundation Models API."""

    def __init__(
        self,
        folder_id: str | None = None,
        iam_token: str | None = None,
        api_key: str | None = None,
        data_logging_enabled: bool = False,
        **kwargs,
    ) -> None:
        """Initialize `BaseYandexFoundationModelsClient`.

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

    def __enter__(self) -> "BaseYandexFoundationModelsClient":
        """Initialize httpx client in the context manager.

        Returns
        -------
        - `BaseYandexFoundationModelsClient`: The client instance
        """
        self._client = Client(headers=self._headers, **self._httpx_client_options)  # type: ignore[reportAny]
        return self

    def __exit__(self, *args, **kwargs) -> None:
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
