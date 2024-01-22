import time

import httpx

from .config import (
    API_URL_OPERATIONS_STATUS,
    API_URL_TEXTGENERATION,
    API_URL_TEXTGENERATION_ASYNC,
)
from .models import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseModel,
    Operation,
)


class YandexLLMClient:
    def __init__(self, iam_token: str, folder_id: str) -> None:
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.api_url = API_URL_TEXTGENERATION
        self.headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id,
        }

    def post_completion(
        self,
        request_data: CompletionRequest,
    ) -> CompletionResponseModel | None:
        with httpx.Client() as client:
            response = client.post(
                self.api_url,
                headers=self.headers,
                json=request_data.model_dump(mode="python"),
                timeout=30,
            )
            if response.status_code == 200:
                return CompletionResponseModel(**response.json())
            else:
                return None

    def post_completion_async(
        self,
        request_data: CompletionRequest,
    ) -> Operation | None:
        with httpx.Client() as client:
            response = client.post(
                API_URL_TEXTGENERATION_ASYNC,
                headers=self.headers,
                json=request_data.model_dump(mode="python"),
                timeout=30,
            )
            if response.status_code == 200:
                return Operation(**response.json())
            else:
                return None

    def get_operation_status(self, operation_id: str) -> Operation | None:
        operation_status_url = API_URL_OPERATIONS_STATUS.format(
            operation_id=operation_id,
        )
        with httpx.Client() as client:
            response = client.get(
                operation_status_url,
                headers=self.headers,
                timeout=30,
            )
            if response.status_code == 200:
                return Operation(**response.json())
            else:
                return None

    def wait_for_completion(
        self,
        operation_id: str,
        poll_interval: float = 1.0,
    ) -> CompletionResponse | None:
        while True:
            operation = self.get_operation_status(operation_id)
            if operation is None:
                return None
            if operation.done:
                if hasattr(operation, "response") and operation.response is not None:
                    return operation.response
                elif hasattr(operation, "error"):
                    return None
            time.sleep(poll_interval)
