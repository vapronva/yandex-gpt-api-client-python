from datetime import datetime
from enum import StrEnum, auto

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class AlternativeStatus(StrEnum):
    ALTERNATIVE_STATUS_UNSPECIFIED = auto()
    ALTERNATIVE_STATUS_PARTIAL = auto()
    ALTERNATIVE_STATUS_TRUNCATED_FINAL = auto()
    ALTERNATIVE_STATUS_FINAL = auto()

    @classmethod
    def _missing_(cls, value: str) -> str | None:
        value = value.lower()
        for member in cls:
            if member == value:
                return member
        return None


class MessageRole(StrEnum):
    SYSTEM = auto()
    ASSISTANT = auto()
    USER = auto()


class Message(BaseModel):
    role: MessageRole
    text: str


class CompletionOptions(BaseModel):
    stream: bool = False
    temperature: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.6
    maxTokens: int


class CompletionRequest(BaseModel):
    modelUri: str
    completionOptions: CompletionOptions
    messages: list[Message]


class Alternative(BaseModel):
    message: Message
    status: AlternativeStatus


class Usage(BaseModel):
    inputTextTokens: int
    completionTokens: int
    totalTokens: int


class CompletionResponse(BaseModel):
    alternatives: list[Alternative]
    usage: Usage
    modelVersion: str


class CompletionAPIResponse(BaseModel):
    result: CompletionResponse


class Operation(BaseModel):
    id: str
    description: str
    createdAt: datetime
    createdBy: str
    modifiedAt: datetime
    done: bool
    metadata: dict | None = None
    error: dict | None = None
    response: CompletionResponse | None = None


class TokenizeRequest(BaseModel):
    modelUri: str
    text: str


class Token(BaseModel):
    id: str
    text: str
    special: bool


class TokenizeResponse(BaseModel):
    tokens: list[Token]
    modelVersion: str
