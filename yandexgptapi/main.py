import httpx
from pydantic import BaseModel, Field
from pathlib import Path


class Message(BaseModel):
    role: str
    text: str


class CompletionOptions(BaseModel):
    stream: bool
    temperature: float
    maxTokens: int = Field(alias="maxTokens")


class CompletionRequest(BaseModel):
    modelUri: str
    completionOptions: CompletionOptions
    messages: list[Message]


class Alternative(BaseModel):
    message: Message
    status: str


class Usage(BaseModel):
    inputTextTokens: str
    completionTokens: str
    totalTokens: str


class CompletionResponse(BaseModel):
    alternatives: list[Alternative]
    usage: Usage
    modelVersion: str


class CompletionResponseModel(BaseModel):
    result: CompletionResponse


class YandexLLMClient:
    def __init__(self, iam_token: str, folder_id: str):
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id,
        }

    def post_completion(
        self, request_data: CompletionRequest
    ) -> CompletionResponse | None:
        with httpx.Client() as client:
            response = client.post(
                self.api_url,
                headers=self.headers,
                json=request_data.model_dump(mode="python"),
                timeout=30,
            )
            if response.status_code == 200:
                return CompletionResponseModel(**response.json()).result
            else:
                print(f"Error: {response.status_code}")
                print(response.json())
                return None


# Usage example
if __name__ == "__main__":
    iam_token = ""
    folder_id = ""
    request_payload = CompletionRequest(
        modelUri="gpt:///yandexgpt/latest",
        completionOptions=CompletionOptions(
            stream=False,
            temperature=0.5,
            maxTokens=2048,
        ),
        messages=[
            Message(
                role="system",
                text="""""",
            ),
            Message(
                role="user",
                text="" + Path("input.txt").read_text(),
            ),
        ],
    )
    client = YandexLLMClient(iam_token=iam_token, folder_id=folder_id)
    response = client.post_completion(request_data=request_payload)
    if response:
        print(response.model_dump_json())
        with open(Path("output.txt"), "w") as f:
            f.write(response.alternatives[0].message.text)
