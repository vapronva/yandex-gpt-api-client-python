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
    operation = client.post_completion_async(request_data=request_payload)
    if operation:
        print(f"Operation ID: {operation.id}")
        response = client.wait_for_completion(operation.id)
        if response:
            print(response.alternatives[0].message.text)
        else:
            print("Failed to get a response or an error occurred.")
