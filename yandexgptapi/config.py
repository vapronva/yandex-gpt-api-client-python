from enum import StrEnum


class ModelURI(StrEnum):
    YANDEX_GPT = "yandexgpt/latest"
    YANDEX_GPT_LITE = "yandexgpt-lite/latest"
    SUMMARIZATION = "summarization/latest"


class API_URLS(StrEnum):
    TEXTGENERATION = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    TEXTGENERATION_ASYNC = "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync"
    OPERATIONS = "https://llm.api.cloud.yandex.net/operations/{operation_id}"
