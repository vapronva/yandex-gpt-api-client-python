import json
import time
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
    def __init__(
        self,
        folder_id: str | None = None,
        iam_token: str | None = None,
        api_key: str | None = None,
        data_logging_enabled: bool = False,
        **kwargs,
    ) -> None:
        if not iam_token and not api_key:
            raise ValueError("Either iam_token or api_key must be provided")
        if not folder_id and iam_token:
            raise ValueError("folder_id is required when using iam_token")
        if iam_token and api_key:
            raise ValueError("Only one of iam_token or api_key must be provided")
        self._headers: dict[str, str] = {
            "x-folder-id": f"{folder_id}",
            "x-data-logging-enabled": "true" if data_logging_enabled else "false",
            "Authorization": f"Api-Key {api_key}" if api_key else f"Bearer {iam_token}",
        }
        if api_key and not folder_id:
            self._headers.pop("x-folder-id")
        self._httpx_client_options = kwargs or {}

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

    def _make_request(
        self,
        method: str,
        url: str,
        request_data: BaseModel | None = None,
    ) -> Response:
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
        request_args: dict[str, Any] = {"url": url}
        if request_data:
            request_args["json"] = request_data.model_dump(mode="python")
        with self._client.stream(method, **request_args) as response:
            yield from response.iter_text()

    def post_completion(
        self,
        request_data: CompletionRequest,
    ) -> CompletionResponse:
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
        request_data.completionOptions.stream = True
        response: Generator[str, None, None] = self._make_stream_request(
            method="post",
            url=APIEndpointsV1.TEXT_GENERATION,
            request_data=request_data,
        )
        for chunk in response:
            parsed_response = CompletionAPIResponse(**json.loads(chunk))
            yield parsed_response.result

    def post_completion_async(self, request_data: CompletionRequest) -> Operation:
        response: Response = self._make_request(
            method="post",
            url=APIEndpointsV1.TEXT_GENERATION_ASYNC,
            request_data=request_data,
        )
        return Operation(**response.json())

    def get_operation_status(self, operation_id: str) -> Operation:
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

    def post_tokenize(self, request_data: TokenizeRequest) -> TokenizeResponse:
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
        response: Response = self._make_request(
            method="post",
            url=APIEndpointsV1.TOKENIZE_COMPLETION,
            request_data=request_data,
        )
        return TokenizeResponse(**response.json())
