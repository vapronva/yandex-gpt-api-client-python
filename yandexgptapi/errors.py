from typing import Any


class BaseYandexGPTAPIError(Exception):
    """Base class for YandexGPT API errors.

    Args
    ----
        - `grpc_code` (`int`): gRPC status code.
        - `http_code` (`int`, optional): HTTP status code.
        - `message` (`str`): Error message.
        - `details` (`list[str]`): Error details.
        - `solution` (`str`, optional): Proposed solution for the error.
    """

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


class ProhibitedTopicError(BaseYandexGPTAPIError):
    """Error raised when a prohibited topic is encountered."""

    def __init__(self) -> None:
        super().__init__(
            grpc_code=3,
            http_code=None,
            message="An answer to a give topic cannot be generated",
            details=[],
            solution=None,
        )


class QuotaExceededError(BaseYandexGPTAPIError):
    """Error raised when the quota is exceeded."""

    def __init__(self) -> None:
        super().__init__(
            grpc_code=8,
            http_code=429,
            message="Quota exceeded",
            details=[],
            solution="Depending on the quota, wait or contact technical support to increase the quota",
        )
