from datetime import datetime
from enum import StrEnum, auto

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class AlternativeStatus(StrEnum):
    """The status of the "alternative" (a term used by Yandex for "choice" in the context of OpenAI schemas). Used in streaming responses to indicate the status of the alternative.

    Attributes
    ----------
        ALTERNATIVE_STATUS_UNSPECIFIED: The status of the alternative is unspecified.
        ALTERNATIVE_STATUS_PARTIAL: The alternative is a partial response.
        ALTERNATIVE_STATUS_TRUNCATED_FINAL: The alternative is a final response, but it is truncated.
        ALTERNATIVE_STATUS_FINAL: The alternative is a final response.

    """

    ALTERNATIVE_STATUS_UNSPECIFIED = auto()
    ALTERNATIVE_STATUS_PARTIAL = auto()
    ALTERNATIVE_STATUS_TRUNCATED_FINAL = auto()
    ALTERNATIVE_STATUS_FINAL = auto()

    @classmethod
    def _missing_(cls, value: str) -> str | None:
        value = value.lower()
        return next((member for member in cls if member == value), None)


class MessageRole(StrEnum):
    """The role of the message. Used in the request to indicate the role of the message.

    Attributes
    ----------
        SYSTEM: The message is from the system.
        ASSISTANT: The message is from the assistant.
        USER: The message is from the user.

    """

    SYSTEM = auto()
    ASSISTANT = auto()
    USER = auto()


class Message(BaseModel):
    """A message to be sent to the model. Used in the request to indicate the message to be sent to the model.

    Attributes
    ----------
        role: MessageRole — The role of the message.
        text: str — The text of the message.

    """

    role: MessageRole
    text: str


class CompletionOptions(BaseModel):
    """The options for the text completion request.

    Attributes
    ----------
        stream: bool — Whether to use streaming for the response.
        temperature: float — The temperature of the response. Must be a float between 0 and 1.
        maxTokens: int — The maximum number of tokens to generate. Depends on the model, but 8048 is the maximum for the Yandex GPT models.

    """

    stream: bool = False
    temperature: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.6
    maxTokens: int


class CompletionRequest(BaseModel):
    """The request to be sent to the model for text completion.

    Attributes
    ----------
        modelUri: str — The URI of the model.
        completionOptions: CompletionOptions — The options for the text completion request.
        messages: list[Message] — The messages to be sent to the model.

    """

    modelUri: str
    completionOptions: CompletionOptions
    messages: list[Message]


class Alternative(BaseModel):
    """An alternative (or the main choice) in the response to the text completion request. Used in the response to indicate the alternative (or the main choice) in the response to the text completion request.

    Attributes
    ----------
        message: Message — The message of the alternative.
        status: AlternativeStatus — The status of the alternative.

    """

    message: Message
    status: AlternativeStatus


class Usage(BaseModel):
    """The usage of the model. Used in the response to indicate the usage of the model during the API call.

    Attributes
    ----------
        inputTextTokens: int — The number of tokens in the input text.
        completionTokens: int — The number of tokens in the completion.
        totalTokens: int — The total number of tokens used.

    """

    inputTextTokens: int
    completionTokens: int
    totalTokens: int


class CompletionResponse(BaseModel):
    """The response to the text completion request.

    Attributes
    ----------
        alternatives: list[Alternative] — The alternatives (or the main choices) in the response to the text completion request.
        usage: Usage — The usage of the model during the API call.
        modelVersion: str — The version of the model used (a date in the "DD.MM.YYYY" format).

    """

    alternatives: list[Alternative]
    usage: Usage
    modelVersion: str


class CompletionAPIResponse(BaseModel):
    """The response to the text completion request from the API.

    Attributes
    ----------
        result: CompletionResponse — The result of the text completion request.

    """

    result: CompletionResponse


class Operation(BaseModel):
    """The asynchronous operation for the text completion request.

    Attributes
    ----------
        id: str — The ID of the operation.
        description: str — The description of the operation.
        createdAt: datetime — The date and time when the operation was created.
        createdBy: str — The user who created the operation.
        modifiedAt: datetime — The date and time when the operation was last modified.
        done: bool — Whether the operation is done.
        metadata: dict | None — The metadata of the operation.
        error: dict | None — The error of the operation.
        response: CompletionResponse | None — The response of the operation.

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
    """The request to be sent for tokenization.

    Attributes
    ----------
        modelUri: str — The URI of the model.
        text: str — The text to be tokenized.

    """

    modelUri: str
    text: str


class Token(BaseModel):
    """A single token representation.

    Attributes
    ----------
        id: str — The ID of the token.
        text: str — The text of the token.
        special: bool — Whether the token is special.

    """

    id: str
    text: str
    special: bool


class TokenizeResponse(BaseModel):
    """The response to the tokenization request.

    Attributes
    ----------
        tokens: list[Token] — The tokens of the text.
        modelVersion: str — The version of the model used (a date in the "DD.MM.YYYY" format).

    """

    tokens: list[Token]
    modelVersion: str
