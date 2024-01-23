from yandexgptapi.config import ModelURI
from yandexgptapi.models import CompletionOptions, CompletionRequest, Message
from yandexgptapi.yandex_llm_client import YandexLLMClient

if __name__ == "__main__":
    iam_token = "your-iam-token"
    folder_id = "your-folder-id"
    request_payload = CompletionRequest(
        modelUri=f"gpt://{folder_id}/{ModelURI.YANDEX_GPT.value}",
        completionOptions=CompletionOptions(
            stream=False,
            temperature=0.5,
            maxTokens=2048,
        ),
        messages=[
            Message(role="system", text=""),
            Message(role="user", text=""),
        ],
    )
    # TextGeneration
    with YandexLLMClient(
        iam_token=iam_token, folder_id=folder_id, timeout=10
    ) as client:
        client.headers = {"x-data-logging-enabled": "false"}
        response = client.post_completion(request_data=request_payload)
        print(response.alternatives[0].message.text)
    # TextGenerationAsync
    with YandexLLMClient(
        iam_token=iam_token, folder_id=folder_id, timeout=10
    ) as client:
        client.headers = {"x-data-logging-enabled": "false"}
        operation = client.post_completion_async(request_data=request_payload)
        if operation:
            print(f"Operation ID: {operation.id}")
            response = client.wait_for_completion(operation.id)
            print(response.alternatives[0].message.text)
