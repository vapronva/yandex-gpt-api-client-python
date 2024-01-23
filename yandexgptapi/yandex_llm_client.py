import time

from httpx import Client, Response

from .config import API_URLS
from .models import (
    CompletionAPIResponse,
    CompletionRequest,
    CompletionResponse,
    Operation,
)


class YandexLLMClient:
    def __init__(
        self,
        iam_token: str,
        folder_id: str,
        data_logging_enabled: bool = False,
        **kwargs,
    ) -> None:
        self._headers: dict[str, str] = {
            "Authorization": f"Bearer {iam_token}",
            "x-folder-id": f"{folder_id}",
            "x-data-logging-enabled": "false" if not data_logging_enabled else "true",
        }
        self._httpx_client_options = kwargs

    def __enter__(self) -> "YandexLLMClient":
        self._client = Client(headers=self._headers, **self._httpx_client_options)
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._client.close()

    @property
    def headers(self) -> dict:
        return self._headers

    @headers.setter
    def headers(self, new_headers: dict[str, str]) -> None:
        self._headers = new_headers
        self._client.headers.update(new_headers)

    def post_completion(self, request_data: CompletionRequest) -> CompletionResponse:
        response: Response = self._client.post(
            url=API_URLS.TEXTGENERATION,
            json=request_data.model_dump(mode="python"),
        )
        response.raise_for_status()
        parsed_response = CompletionAPIResponse(**response.json())
        return parsed_response.result

    def post_completion_async(self, request_data: CompletionRequest) -> Operation:
        response: Response = self._client.post(
            url=API_URLS.TEXTGENERATION_ASYNC,
            json=request_data.model_dump(mode="python"),
        )
        response.raise_for_status()
        return Operation(**response.json())

    def get_operation_status(self, operation_id: str) -> Operation:
        response: Response = self._client.get(
            url=API_URLS.OPERATIONS.format(operation_id=operation_id),
        )
        response.raise_for_status()
        return Operation(**response.json())

    def wait_for_completion(
        self,
        operation_id: str,
        poll_interval: float = 1.0,
    ) -> CompletionResponse:
        while True:
            operation: Operation = self.get_operation_status(operation_id)
            if operation.done and operation.response:
                return operation.response
            elif operation.error:
                msg: str = (
                    f"Operation #{operation_id} failed with error: {operation.error}"
                )
                raise RuntimeError(msg)
            time.sleep(poll_interval)
