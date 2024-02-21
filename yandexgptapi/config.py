from enum import StrEnum


class GenerativeModelURI(StrEnum):
    """Enum for URIs of generative models.

    Attributes
    ----------
        - `YANDEX_GPT`: URI for Yandex GPT model.
        - `YANDEX_GPT_LITE`: URI for Yandex GPT Lite model.
        - `SUMMARIZATION`: URI for summarization model.
        - `FINETUNED_DATASPHERE`: URI for finetuned DataSphere model.
    """

    YANDEX_GPT = "gpt://{folder_id}/yandexgpt/latest"
    YANDEX_GPT_LITE = "gpt://{folder_id}/yandexgpt-lite/latest"
    SUMMARIZATION = "gpt://{folder_id}/summarization/latest"
    FINETUNED_DATASPHERE = "ds://{fn_model_id}"


class EmbeddingModelURI(StrEnum):
    """Enum for URIs of embedding models.

    Attributes
    ----------
        - `TEXT_EMBEDDING`: URI for text embedding model for search queries.
        - `EMBEDDING_SEARCH_QUERY`: URI for text embedding model for knowledge bases and documentation.
    """

    EMBEDDING_SEARCH_QUERY = "emb://{folder_id}/text-search-query/latest"
    EMBEDDING_SEARCH_DOC = "emb://{folder_id}/text-search-doc/latest"


class APIEndpointsV1(StrEnum):
    """Enum for endpoints of Yandex GPT API.

    Attributes
    ----------
        - `TEXT_GENERATION`: Endpoint for text generation.
        - `TEXT_GENERATION_ASYNC`: Endpoint for asynchronous text generation.
        - `OPERATIONS`: Endpoint for operations.
        - `TOKENIZE`: Endpoint for tokenization.
        - `TOKENIZE_COMPLETION`: Endpoint for tokenization completion.
        - `TEXT_EMBEDDING`: Endpoint for text embedding.
    """

    TEXT_GENERATION = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    TEXT_GENERATION_ASYNC = "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync"
    OPERATIONS = "https://llm.api.cloud.yandex.net/operations/{operation_id}"
    TOKENIZE = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenize"
    TOKENIZE_COMPLETION = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenizeCompletion"
    TEXT_EMBEDDING = "https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding"
