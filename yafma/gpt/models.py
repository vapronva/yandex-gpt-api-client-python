from datetime import datetime
from enum import StrEnum, auto
from typing import Any, override

from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt
from typing_extensions import Annotated

from .config import GenerativeModelURI


class AlternativeStatus(StrEnum):
    """Enum representing a generation status of the alternative in a response.

    Attributes
    ----------
    - `ALTERNATIVE_STATUS_UNSPECIFIED`: Unspecified status
    - `ALTERNATIVE_STATUS_PARTIAL`: Partially generated response
    - `ALTERNATIVE_STATUS_TRUNCATED_FINAL`: Final but truncated response (max tokens reached)
    - `ALTERNATIVE_STATUS_FINAL`: Final response without any errors or running into any limits
    - `ALTERNATIVE_STATUS_CONTENT_FILTER`: Response was filtered/stopped due to the potentiallty sensitive content in the prompt or response

    Notes
    -----
    - "alternative" is a term used in the Yandex, which is simillar to "choices" in OpenAI's API.

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#responses](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#responses)
    """

    ALTERNATIVE_STATUS_UNSPECIFIED = auto()
    ALTERNATIVE_STATUS_PARTIAL = auto()
    ALTERNATIVE_STATUS_TRUNCATED_FINAL = auto()
    ALTERNATIVE_STATUS_FINAL = auto()
    ALTERNATIVE_STATUS_CONTENT_FILTER = auto()

    @override
    @classmethod
    def _missing_(cls, value: object) -> str | None:
        if isinstance(value, str):
            value = value.lower()
        return next((member for member in cls if member == value), None)


class MessageRole(StrEnum):
    """Enum representing a role of the message in a request.

    Attributes
    ----------
    - `SYSTEM`: Defines the behaviour of the model
    - `ASSISTANT`: Model's response
    - `USER`: Main request to the model

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#body_params](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#body_params)
    """

    SYSTEM = auto()
    ASSISTANT = auto()
    USER = auto()


class Message(BaseModel):
    """Model representing a message to be sent to a model.

    Attributes
    ----------
    - `role` (`MessageRole`): Message role
    - `text` (`str`): Text content

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#body_params](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#body_params)
    """

    role: MessageRole
    text: str


class CompletionOptions(BaseModel):
    """Model representing the options for a text completion request.

    Attributes
    ----------
    - `stream` (`bool`): Enable streaming of partially generated text
    - `temperature` (`float`): Sampling temperature (a double number between 0 and 1)
    - `maxTokens` (`int`): Maximum number of tokens to generate

    Notes
    -----
    - `temperature` is a hyperparameter controlling the randomness of the generated text. Lower values make the text more deterministic, while higher values make it more random and "creative".

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#body_params](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#body_params)
    """

    stream: bool = False
    temperature: Annotated[float, Field(strict=True, ge=0, le=1)] = 0.6
    maxTokens: PositiveInt | None = None


class CompletionRequest(BaseModel):
    """Model representing the request for text completion.

    Attributes
    ----------
    - `modelUri` (`str` or `GenerativeModelURI`): ID of the model to use
    - `completionOptions` (`CompletionOptions`): Configuration options for the completion generation
    - `messages` (`list[Message]`): List of messages (context for the completion)

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#body_params](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#body_params)
    """

    modelUri: str | GenerativeModelURI = GenerativeModelURI.PRO
    completionOptions: CompletionOptions = CompletionOptions()
    messages: list[Message]


class Alternative(BaseModel):
    """Model representing the alternative in a text completion response.

    Attributes
    ----------
    - `message` (`Message`): Message object representing the alternative
    - `status` (`AlternativeStatus`): Status of the alternative

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#responses](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#responses)
    """

    message: Message
    status: AlternativeStatus


class Usage(BaseModel):
    """Model representing the usage of a model during the API call.

    Attributes
    ----------
    - `inputTextTokens` (`NonNegativeInt`): Number of tokens in the input text
    - `completionTokens` (`NonNegativeInt`): Number of tokens in the generated completions
    - `totalTokens` (`NonNegativeInt`): Total number of tokens used

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#responses](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#responses)
    """

    inputTextTokens: NonNegativeInt
    completionTokens: NonNegativeInt
    totalTokens: NonNegativeInt


class CompletionResponse(BaseModel):
    """Model representing the response to the text completion request.

    Attributes
    ----------
    - `alternatives` (`list[Alternative]`): Generated completion alternatives
    - `usage` (`Usage`): Model usage during the API call
    - `modelVersion` (`str`): Version of the model used (a date in `DD.MM.YYYY` format)

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#responses](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGeneration/completion#responses)
    """

    alternatives: list[Alternative]
    usage: Usage
    modelVersion: str


class CompletionAPIResponse(BaseModel):
    """Model representing the response from the API to the text completion request.

    Attributes
    ----------
    - `result` (`CompletionResponse`): Result of the text completion request
    """

    result: CompletionResponse


class GrpcStatus(BaseModel):
    """Model representing a gRPC-compliant status of an operation.

    Attributes
    ----------
    - `code` (`int`): Status code
    - `message` (`str`): Error message
    - `details` (`list[Any]`, optional): Error details

    Implements
    ----------
    [cloud.google.com/tasks/docs/reference/rpc/google.rpc#status](https://cloud.google.com/tasks/docs/reference/rpc/google.rpc#status)
    """

    code: int
    message: str
    details: list[Any] = []


class Operation(BaseModel):
    """Model representing the asynchronous operation for a text completion request.

    Attributes
    ----------
    - `id` (`str`): Operation ID (UUID)
    - `description` (`str`): Operation description
    - `createdAt` (`datetime`): Creation timestamp
    - `createdBy` (`str`): Creator (user or service account ID) of the operation
    - `modifiedAt` (`datetime`): Last modification timestamp
    - `done` (`bool`): Operation completion status
    - `metadata` (`dict[str, Any]`, optional): Any additional metadata about the operation
    - `error` (`dict`, optional): Error result
    - `response` (`CompletionResponse`, optional): Operation response

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGenerationAsync/completion#responses](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/TextGenerationAsync/completion#responses)

    Notes
    -----
    - If the operation was not successful (failed) or was cancelled, the `error` field will be present and `response` will not (and vice versa).
    - `done` is `False` if the operation is still in progress, and `True` if it is completed (either successfully or not).
    """

    id: str
    description: str
    createdAt: datetime
    createdBy: str
    modifiedAt: datetime
    done: bool
    metadata: dict[str, Any] | None = None
    error: GrpcStatus | None = None
    response: CompletionResponse | None = None


class TokenizeRequest(BaseModel):
    """Model representing the tokenization request.

    Attributes
    ----------
    - `modelUri` (`str` or `GenerativeModelURI`): ID of the model to use
    - `text` (`str`): Text to tokenize

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenize#https-request](https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenize#https-request)
    """

    modelUri: GenerativeModelURI | str = GenerativeModelURI.PRO
    text: str


class Token(BaseModel):
    """Model representing the single token of a language model.

    Attributes
    ----------
    - `id` (`str`): Token ID
    - `text` (`str`): Text content of the token
    - `special` (`bool`): Whether the token is special (e.g., a separator or a role marker)

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenize#responses][https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenize#responses]
    """

    id: str
    text: str
    special: bool


class TokenizeResponse(BaseModel):
    """Model representing the response to the tokenization request.

    Attributes
    ----------
    - `tokens` (`list[Token]`): List of tokens (tokenized text)
    - `modelVersion` (`str`): Version of the model used (a date in `DD.MM.YYYY` format)

    Implements
    ----------
    [yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenize#responses][https://yandex.cloud/en/docs/foundation-models/text-generation/api-ref/Tokenizer/tokenize#responses]
    """

    tokens: list[Token]
    modelVersion: str
