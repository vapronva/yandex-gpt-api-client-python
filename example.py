import os

from yandexgptapi.config import ModelURI
from yandexgptapi.models import (
    CompletionOptions,
    CompletionRequest,
    CompletionResponse,
    Message,
    Operation,
    TokenizeRequest,
    TokenizeResponse,
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
        # TextGeneration
        response_tg: CompletionResponse = client.post_completion(
            request_data=request_payload,
        )
        print(response_tg.alternatives[0].message.text)
        # TextGenerationAsync
        operation_tga: Operation = client.post_completion_async(
            request_data=request_payload,
        )
        print(f"Operation ID: {operation_tga.id}")
        response_tga: CompletionResponse = client.wait_for_completion(operation_tga.id)
        print(response_tga.alternatives[0].message.text)
        # Tokenize
        response_t: TokenizeResponse = client.post_tokenize(
            request_data=TokenizeRequest(
                modelUri=f"gpt://{folder_id}/{ModelURI.YANDEX_GPT.value}",
                text="Ты - Саратов",
            ),
        )
        print(response_t.tokens)
        # TokenizeCompletion
        response_tc: TokenizeResponse = client.post_tokenize_completion(
            request_data=request_payload,
        )
        print(response_tc.tokens)
