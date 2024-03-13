from datetime import datetime
from enum import StrEnum, auto

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class AlternativeStatus(StrEnum):
    """Enum representing the status of an "alternative" in streaming responses.
    Note: "alternative" is a term used in the Yandex, which is simillar to "choices" in OpenAI API.

    Attributes
    ----------
        - `ALTERNATIVE_STATUS_UNSPECIFIED`: Unspecified status.
        - `ALTERNATIVE_STATUS_PARTIAL`: Partial response.
        - `ALTERNATIVE_STATUS_TRUNCATED_FINAL`: Final but truncated response.
        - `ALTERNATIVE_STATUS_FINAL`: Final response.
        - `ALTERNATIVE_STATUS_CONTENT_FILTER`: Response was filtered.
    """

    ALTERNATIVE_STATUS_UNSPECIFIED = auto()
    ALTERNATIVE_STATUS_PARTIAL = auto()
    ALTERNATIVE_STATUS_TRUNCATED_FINAL = auto()
    ALTERNATIVE_STATUS_FINAL = auto()
    ALTERNATIVE_STATUS_CONTENT_FILTER = auto()

    @classmethod
    def _missing_(cls, value: str) -> str | None:
        value = value.lower()
        return next((member for member in cls if member == value), None)


class MessageRole(StrEnum):
    """Enum representing the role of a message in a request.

    Attributes
    ----------
        - `SYSTEM`: Message from the system (defines the behaviour of the model).
        - `ASSISTANT`: Message from the assistant (model's response).
        - `USER`: Message from the user (main request to the model).
    """

    SYSTEM = auto()
    ASSISTANT = auto()
    USER = auto()


class Message(BaseModel):
    """Model representing a message to be sent to the model.

    Attributes
    ----------
        - `role` (`MessageRole`): Role of the message .
        - `text` (`str`): Text of the message.
    """

    role: MessageRole
    text: str


class CompletionOptions(BaseModel):
    """Model representing options for a text completion request.

    Attributes
    ----------
        - `stream` (`bool`): Whether to use streaming for the response.
        - `temperature` (`float`): Temperature of the response, between 0 and 1.
        - `maxTokens` (`int`): Maximum number of tokens to generate.
    """

    stream: bool = False
    temperature: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.6
    maxTokens: int


class CompletionRequest(BaseModel):
    """Model representing a request for text completion.

    Attributes
    ----------
        - `modelUri` (`str`): URI of the model.
        - `completionOptions` (`CompletionOptions`): Options for the text completion request.
        - `messages` (`list[Message]`): Messages to be sent to the model.
    """

    modelUri: str
    completionOptions: CompletionOptions
    messages: list[Message]


class Alternative(BaseModel):
    """Model representing an alternative in a text completion response.

    Attributes
    ----------
        - `message` (`Message`): Message of the alternative.
        - `status` (`AlternativeStatus`): Status of the alternative.
    """

    message: Message
    status: AlternativeStatus


class Usage(BaseModel):
    """Model representing the usage of the model during an API call.

    Attributes
    ----------
        - `inputTextTokens` (`int`): Number of tokens in the input text.
        - `completionTokens` (`int`): Number of tokens in the completion.
        - `totalTokens` (`int`): Total number of tokens used.
    """

    inputTextTokens: int
    completionTokens: int
    totalTokens: int


class CompletionResponse(BaseModel):
    """Model representing a response to a text completion request.

    Attributes
    ----------
        - `alternatives` (`list[Alternative]`): Alternatives in the response.
        - `usage` (`Usage`): Usage of the model during the API call.
        - `modelVersion` (`str`): Version of the model used (a date in `DD.MM.YYYY` format).
    """

    alternatives: list[Alternative]
    usage: Usage
    modelVersion: str


class CompletionAPIResponse(BaseModel):
    """Model representing a response from the API to a text completion request.

    Attributes
    ----------
        - `result` (`CompletionResponse`): Result of the text completion request.
    """

    result: CompletionResponse


class Operation(BaseModel):
    """Model representing an asynchronous operation for a text completion request.

    Attributes
    ----------
        - `id` (`str`): Operation ID.
        - `description` (`str`): Operation description.
        - `createdAt` (`datetime`): Creation timestamp.
        - `createdBy` (`str`): Creator of the operation.
        - `modifiedAt` (`datetime`): Last modification timestamp.
        - `done` (`bool`): Operation completion status.
        - `metadata` (`dict`, optional): Operation metadata.
        - `error` (`dict`, optional): Operation error.
        - `response` (`CompletionResponse`, optional): Operation response.
    """

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
    """Model representing a tokenization request.

    Attributes
    ----------
        - `modelUri` (`str`): Model URI.
        - `text` (`str`): Text to tokenize.
    """

    modelUri: str
    text: str


class Token(BaseModel):
    """Model representing a single token.

    Attributes
    ----------
        - `id` (`str`): Token ID.
        - `text` (`str`): Text of the token.
        - `special` (`bool`): Whether the token is special.
    """

    id: str
    text: str
    special: bool


class TokenizeResponse(BaseModel):
    """Model representing a response to a tokenization request.

    Attributes
    ----------
        - `tokens` (`list[Token]`): Tokens in the response.
        - `modelVersion` (`str`): Version of the model used (a date in `DD.MM.YYYY` format).
    """

    tokens: list[Token]
    modelVersion: str
