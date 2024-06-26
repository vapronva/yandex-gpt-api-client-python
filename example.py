from os import getenv
from time import sleep

from pydantic import NonNegativeFloat, PositiveInt

from yafma import EmbeddingsClient, YandexGptClient
from yafma.embedding import (
    EmbeddingModelURI,
    EmbeddingsRequest,
    EmbeddingsResponse,
)
from yafma.gpt import (
    CompletionOptions,
    CompletionRequest,
    CompletionResponse,
    GenerativeModelURI,
    Message,
    MessageRole,
    TokenizeRequest,
    TokenizeResponse,
)
from yafma.gpt.models import Operation

WAIT_TIME: NonNegativeFloat = 1.5
TEMPERATURE: NonNegativeFloat = 0.6
MAX_TOKENS: PositiveInt = 256
TIMEOUT: NonNegativeFloat = 10.0

YC_IAM_TOKEN: str = getenv("YANDEX_CLOUD_IAM_TOKEN", "")
YC_API_KEY: str = getenv("YANDEX_CLOUD_API_KEY", "")
YC_FOLDER_ID: str = getenv("YANDEX_CLOUD_FOLDER_ID", "")


def pr_resp_wait(response: object) -> None:
    print(f"{type(response)}: {response}")
    sleep(WAIT_TIME)


def print_section(section_name: str) -> None:
    print(f"{section_name:=^50}")


def test_all_gpt_api_endpoints(
    client: YandexGptClient,
    request_payload: CompletionRequest,
) -> None:
    print_section("TextGeneration")
    response: CompletionResponse = client.post_completion(
        request_data=request_payload,
    )
    pr_resp_wait(response)
    print_section("TextGeneration (stream)")
    request_payload.completionOptions.stream = True
    response_stream = client.post_completion_stream(request_data=request_payload)
    for chunk_response in response_stream:
        print(chunk_response.alternatives[0].message.text)
    request_payload.completionOptions.stream = False
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
            modelUri=GenerativeModelURI.PRO.format(
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


def test_all_embedding_api_endpoints(
    client: EmbeddingsClient,
    request_payload: EmbeddingsRequest,
) -> None:
    print_section("Embeddings")
    response: EmbeddingsResponse = client.post_embedding(
        request_data=request_payload,
    )
    pr_resp_wait(response)


def main() -> None:
    request_payload = CompletionRequest(
        modelUri=GenerativeModelURI.LITE.format(
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
    with YandexGptClient(api_key=YC_API_KEY) as client:
        print_section("TextGeneration (API Key)")
        response: CompletionResponse = client.post_completion(
            request_data=request_payload,
        )
        pr_resp_wait(response)
    print_section("Auth with IAM Token")
    with YandexGptClient(
        folder_id=YC_FOLDER_ID,
        iam_token=YC_IAM_TOKEN,
        timeout=10,
    ) as client:
        test_all_gpt_api_endpoints(
            client=client,
            request_payload=request_payload,
        )
    with EmbeddingsClient(
        folder_id=YC_FOLDER_ID,
        iam_token=YC_IAM_TOKEN,
        timeout=10,
    ) as client:
        test_all_embedding_api_endpoints(
            client=client,
            request_payload=EmbeddingsRequest(
                modelUri=EmbeddingModelURI.QUERY.format(
                    folder_id=YC_FOLDER_ID,
                ),
                text="Ты - Саратов",
            ),
        )


if __name__ == "__main__":
    main()
