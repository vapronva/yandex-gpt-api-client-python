from pydantic import BaseModel


class Message(BaseModel):
    role: str
    text: str


class CompletionOptions(BaseModel):
    stream: bool
    temperature: float
    maxTokens: int


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


class Operation(BaseModel):
    id: str
    description: str
    createdAt: str
    createdBy: str
    modifiedAt: str
    done: bool
    metadata: dict | None = None
    error: dict | None = None
    response: CompletionResponse | None = None
