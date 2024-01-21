from yandexgptapi.models import CompletionOptions, CompletionRequest, Message
from yandexgptapi.yandex_llm_client import YandexLLMClient

if __name__ == "__main__":
    iam_token = "your-iam-token"
    folder_id = "your-folder-id"
    request_payload = CompletionRequest(
        modelUri="gpt://your-folder-id/yandexgpt/latest",
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
    client = YandexLLMClient(iam_token=iam_token, folder_id=folder_id)
    response = client.post_completion(request_data=request_payload)
    print(response.result.alternatives[0].message.text) if response else None
