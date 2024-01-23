import os

from yandexgptapi.config import ModelURI
from yandexgptapi.models import (
    CompletionOptions,
    CompletionRequest,
    CompletionResponse,
    Message,
    Operation,
)
from yandexgptapi.yandex_llm_client import YandexLLMClient

if __name__ == "__main__":
    iam_token: str = os.getenv("YANDEX_CLOUD_IAM_TOKEN", "")
    folder_id: str = os.getenv("YANDEX_CLOUD_FOLDER_ID", "")
    request_payload = CompletionRequest(
        modelUri=f"gpt://{folder_id}/{ModelURI.YANDEX_GPT.value}",
        completionOptions=CompletionOptions(
            stream=False,
            temperature=0.6,
            maxTokens=256,
        ),
        messages=[
            Message(role="system", text="Ты - Саратов"),
            Message(role="user", text="Кто?"),
        ],
    )
    # TextGeneration
    with YandexLLMClient(
        iam_token=iam_token,
        folder_id=folder_id,
        timeout=10,
    ) as client:
        response: CompletionResponse = client.post_completion(
            request_data=request_payload,
        )
        print(response.alternatives[0].message.text)
    # TextGenerationAsync
    with YandexLLMClient(
        iam_token=iam_token,
        folder_id=folder_id,
        timeout=10,
    ) as client:
        operation: Operation = client.post_completion_async(
            request_data=request_payload,
        )
        print(f"Operation ID: {operation.id}")
        response: CompletionResponse = client.wait_for_completion(operation.id)
        print(response.alternatives[0].message.text)
