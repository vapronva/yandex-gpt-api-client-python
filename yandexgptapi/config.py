from enum import StrEnum


class GenerativeModelURI(StrEnum):
    """The URI for the generative model.

    Attributes
    ----------
        YANDEX_GPT: The URI for the Yandex GPT model.
        YANDEX_GPT_LITE: The URI for the Yandex GPT Lite model.
        SUMMARIZATION: The URI for the summarization model (based on the Yandex GPT Lite model).
        FINETUNED_DATASPHERE: The URI for the finetuned in the DataSphere model.

    """

    YANDEX_GPT = "gpt://{folder_id}/yandexgpt/latest"
    YANDEX_GPT_LITE = "gpt://{folder_id}/yandexgpt-lite/latest"
    SUMMARIZATION = "gpt://{folder_id}/summarization/latest"
    FINETUNED_DATASPHERE = "ds://{fn_model_id}"


class EmbeddingModelURI(StrEnum):
    """The URI for the embedding model.

    Attributes
    ----------
        TEXT_EMBEDDING: The URI for the text embedding model (for search queries).
        EMBEDDING_SEARCH_QUERY: The URI for the text embedding model (for knowledge bases and other documentation).

    """

    EMBEDDING_SEARCH_QUERY = "emb://{folder_id}/text-search-query/latest"
    EMBEDDING_SEARCH_DOC = "emb://{folder_id}/text-search-doc/latest"


class APIEndpointsV1(StrEnum):
    """The API endpoints for the Yandex GPT API."""

    TEXT_GENERATION = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    TEXT_GENERATION_ASYNC = "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync"
    OPERATIONS = "https://llm.api.cloud.yandex.net/operations/{operation_id}"
    TOKENIZE = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenize"
    TOKENIZE_COMPLETION = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenizeCompletion"
    TEXT_EMBEDDING = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
