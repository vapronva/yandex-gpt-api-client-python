from typing import Any


class BaseYandexGPTAPIError(Exception):
    def __init__(
        self,
        grpc_code: int,
        http_code: int | None,
        message: str,
        details: list[str],
        solution: str | None,
    ) -> None:
        self.grpc_code: int = grpc_code
        self.http_code: int | None = http_code
        self.message: str = message
        self.details: list[Any] = details
        self.solution: str | None = solution

    def __str__(self) -> str:
        return f"{self.grpc_code=} {self.http_code=} {self.message=} {self.details=} {self.solution=}"


# this is not documented anywhere in the yandex cloud docs, but it did show up once in my testing, and i couldn't replicate it ever since (did not try edgy topics, though)
class ProhibitedTopicError(BaseYandexGPTAPIError):
    def __init__(self) -> None:
        super().__init__(
            grpc_code=3,
            http_code=None,
            message="An answer to a give topic cannot be generated",
            details=[],
            solution=None,
        )


class QuotaExceededError(BaseYandexGPTAPIError):
    def __init__(self) -> None:
        super().__init__(
            grpc_code=8,
            http_code=429,
            message="Quota exceeded",
            details=[],
            solution="Depending on the quota, wait or contact technical support to increase the quota",
        )
