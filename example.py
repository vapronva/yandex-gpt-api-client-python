from os import getenv
from time import sleep
from typing import Any

from pydantic import NonNegativeFloat, PositiveInt

from yafma import YandexGPTClient
from yafma.config import GenerativeModelURI
from yafma.models import (
    CompletionOptions,
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    Operation,
    TokenizeRequest,
    TokenizeResponse,
)


WAIT_TIME: NonNegativeFloat = 1.5
TEMPERATURE: NonNegativeFloat = 0.6
MAX_TOKENS: PositiveInt = 256
TIMEOUT: NonNegativeFloat = 10.0

YC_IAM_TOKEN: str = getenv("YANDEX_CLOUD_IAM_TOKEN", "")
YC_API_KEY: str = getenv("YANDEX_CLOUD_API_KEY", "")
YC_FOLDER_ID: str = getenv("YANDEX_CLOUD_FOLDER_ID", "")


def pr_resp_wait(response: Any) -> None:
    print(f"{type(response)}: {response}")
    sleep(WAIT_TIME)


def print_section(section_name: str) -> None:
    print(f"{section_name:=^50}")


def test_all_api_endpoints(
    client: YandexGPTClient,
    request_payload: CompletionRequest,
) -> None:
    print_section("TextGeneration")
    response: CompletionResponse = client.post_completion(
        request_data=request_payload,
    )
    pr_resp_wait(response)
    print_section("TextGeneration (stream)")
    response_stream = client.post_completion_stream(request_data=request_payload)
    for chunk_response in response_stream:
        print(chunk_response.alternatives[0].message.text)
    sleep(WAIT_TIME)
    print_section("TextGenerationAsync")
    operation: Operation = client.post_completion_async(
        request_data=request_payload,
    )
    print(f"Received opeation ID: {operation.id}")
    response_async: CompletionResponse = client.wait_for_completion(operation.id)
    pr_resp_wait(response_async)
    print_section("Tokenize")
    response_tokenize: TokenizeResponse = client.post_tokenize(
        request_data=TokenizeRequest(
            modelUri=GenerativeModelURI.YANDEX_GPT.value.format(
                folder_id=YC_FOLDER_ID,
            ),
            text="Ты - Саратов",
        ),
    )
    pr_resp_wait(response_tokenize)
    print_section("TokenizeCompletion")
    response_tokenize_completion: TokenizeResponse = client.post_tokenize_completion(
        request_data=request_payload,
    )
    pr_resp_wait(response_tokenize_completion)


def main() -> None:
    request_payload = CompletionRequest(
        modelUri=GenerativeModelURI.YANDEX_GPT_LITE.value.format(
            folder_id=YC_FOLDER_ID,
        ),
        completionOptions=CompletionOptions(
            temperature=TEMPERATURE,
            maxTokens=MAX_TOKENS,
        ),
        messages=[
            Message(role=MessageRole.SYSTEM, text="Ты - Саратов"),
            Message(role=MessageRole.USER, text="Кто?"),
        ],
    )
    print_section("Auth with API Key")
    with YandexGPTClient(api_key=YC_API_KEY) as client:
        print_section("TextGeneration (API Key)")
        response: CompletionResponse = client.post_completion(
            request_data=request_payload,
        )
        pr_resp_wait(response)
    print_section("Auth with IAM Token")
    with YandexGPTClient(
        folder_id=YC_FOLDER_ID,
        iam_token=YC_IAM_TOKEN,
        timeout=10,
    ) as client:
        test_all_api_endpoints(client, request_payload)


if __name__ == "__main__":
    main()
