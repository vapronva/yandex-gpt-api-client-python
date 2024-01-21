import httpx

from .config import API_URL
from .models import CompletionRequest, CompletionResponseModel


class YandexLLMClient:
    def __init__(self, iam_token: str, folder_id: str) -> None:
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.api_url = API_URL
        self.headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "x-folder-id": self.folder_id,
        }

    def post_completion(
        self,
        request_data: CompletionRequest,
    ) -> CompletionResponseModel | None:
        with httpx.Client() as client:
            response = client.post(
                self.api_url,
                headers=self.headers,
                json=request_data.model_dump(mode="python"),
                timeout=30,
            )
            if response.status_code == 200:
                return CompletionResponseModel(**response.json())
            else:
                return None
