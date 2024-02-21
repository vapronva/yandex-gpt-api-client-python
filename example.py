from os import getenv
from time import sleep

from yandexgptapi import YandexLLMClient
from yandexgptapi.config import GenerativeModelURI
from yandexgptapi.models import (
    CompletionOptions,
    CompletionRequest,
    CompletionResponse,
    Message,
    MessageRole,
    Operation,
    TokenizeRequest,
    TokenizeResponse,
)

if __name__ == "__main__":
    iam_token: str = getenv("YANDEX_CLOUD_IAM_TOKEN", "")
    api_key: str = getenv("YANDEX_CLOUD_API_KEY", "")
    folder_id: str = getenv("YANDEX_CLOUD_FOLDER_ID", "")
    request_payload = CompletionRequest(
        modelUri=GenerativeModelURI.YANDEX_GPT_LITE.value.format(folder_id=folder_id),
        completionOptions=CompletionOptions(
            temperature=0.6,
            maxTokens=256,
        ),
        messages=[
            Message(role=MessageRole.SYSTEM, text="Ты - Саратов"),
            Message(role=MessageRole.SYSTEM, text="Кто?"),
        ],
    )
    # auth with api key
    with YandexLLMClient(
        api_key=api_key,
        timeout=10,
    ) as client:
        # TextGeneration
        print("{:=^50}".format("TextGeneration (API Key)"))
        response_tg: CompletionResponse = client.post_completion(
            request_data=request_payload,
        )
        print(response_tg)
        sleep(2)
    # auth with iam token
    with YandexLLMClient(
        iam_token=iam_token,
        folder_id=folder_id,
        timeout=10,
    ) as client:
        # TextGeneration
        print("{:=^50}".format("TextGeneration"))
        response_tg: CompletionResponse = client.post_completion(
            request_data=request_payload,
        )
        print(response_tg)
        sleep(2)
        # TextGeneration (stream)
        print("{:=^50}".format("TextGeneration (stream)"))
        response_tgs = client.post_completion_stream(
            request_data=request_payload,
        )
        for chunk_response in response_tgs:
            print(chunk_response.alternatives[0].message.text)
        sleep(2)
        # TextGenerationAsync
        print("{:=^50}".format("TextGenerationAsync"))
        operation_tga: Operation = client.post_completion_async(
            request_data=request_payload,
        )
        print(f"Operation ID: {operation_tga.id}")
        response_tga: CompletionResponse = client.wait_for_completion(operation_tga.id)
        print(response_tga)
        sleep(2)
        # Tokenize
        print("{:=^50}".format("Tokenize"))
        response_t: TokenizeResponse = client.post_tokenize(
            request_data=TokenizeRequest(
                modelUri=GenerativeModelURI.YANDEX_GPT.value.format(
                    folder_id=folder_id,
                ),
                text="Ты - Саратов",
            ),
        )
        print(response_t)
        sleep(2)
        # TokenizeCompletion
        print("{:=^50}".format("TokenizeCompletion"))
        response_tc: TokenizeResponse = client.post_tokenize_completion(
            request_data=request_payload,
        )
        print(response_tc)
