import time

from httpx import Client

from .config import (
    API_URL_OPERATIONS_STATUS,
    API_URL_TEXTGENERATION,
    API_URL_TEXTGENERATION_ASYNC,
)
from .models import (
    CompletionRequest,
    CompletionResponse,
    CompletionAPIResponse,
    Operation,
)


class YandexLLMClient:
    def __init__(self, iam_token: str, folder_id: str, **kwargs) -> None:
        self.api_url = API_URL_TEXTGENERATION
        self._headers = {
            "Authorization": f"Bearer {iam_token}",
            "x-folder-id": f"{folder_id}",
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
        response = self._client.post(
            self.api_url,
            json=request_data.model_dump(mode="python"),
        )
        response.raise_for_status()
        parsed_response = CompletionAPIResponse(**response.json())
        return parsed_response.result

    def post_completion_async(self, request_data: CompletionRequest) -> Operation:
        response = self._client.post(
            API_URL_TEXTGENERATION_ASYNC,
            json=request_data.model_dump(mode="python"),
            timeout=30,
        )
        response.raise_for_status()
        return Operation(**response.json())

    def get_operation_status(self, operation_id: str) -> Operation:
        operation_status_url = API_URL_OPERATIONS_STATUS.format(
            operation_id=operation_id,
        )
        response = self._client.get(operation_status_url, timeout=30)
        response.raise_for_status()
        return Operation(**response.json())

    def wait_for_completion(
        self,
        operation_id: str,
        poll_interval: float = 1.0,
    ) -> CompletionResponse:
        while True:
            operation = self.get_operation_status(operation_id)
            if operation.done and operation.response:
                return operation.response
            elif operation.error:
                raise RuntimeError(
                    f"Operation #{operation_id} failed with error: {operation.error}"
                )
            time.sleep(poll_interval)
