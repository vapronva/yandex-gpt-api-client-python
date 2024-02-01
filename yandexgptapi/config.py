from enum import StrEnum


class GenerativeModelURI(StrEnum):
    YANDEX_GPT = "gpt://{folder_id}/yandexgpt/latest"
    YANDEX_GPT_LITE = "gpt://{folder_id}/yandexgpt-lite/latest"
    SUMMARIZATION = "gpt://{folder_id}/summarization/latest"
    FINETUNED_DATASPHERE = "ds://{fn_model_id}"


class EmbeddingModelURI(StrEnum):
    EMBEDDING_SEARCH_QUERY = "emb://{folder_id}/text-search-query/latest"
    EMBEDDING_SEARCH_DOC = "emb://{folder_id}/text-search-doc/latest"


class APIEndpointsV1(StrEnum):
    TEXT_GENERATION = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    TEXT_GENERATION_ASYNC = "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync"
    OPERATIONS = "https://llm.api.cloud.yandex.net/operations/{operation_id}"
    TOKENIZE = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenize"
    TOKENIZE_COMPLETION = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenizeCompletion"
    TEXT_EMBEDDING = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
