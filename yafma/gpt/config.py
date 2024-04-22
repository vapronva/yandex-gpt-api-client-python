from enum import Enum, StrEnum

from pydantic import BaseModel, PositiveInt


class GenerativeModelURI(StrEnum):
    """Enum for the URIs of the text generation models.

    Attributes
    ----------
    - `PRO`: YandexGPT Pro (v3)
    - `LITE`: YandexGPT Lite (v2)
    - `SUMMARIZATION`: Text recap (v2)
    - `FINETUNE_DATASPHERE`: Any finetuned model in DataSphere (synchronous only) (v3)

    Docs
    ----
    [yandex.cloud/en/docs/foundation-models/concepts/yandexgpt/models](https://yandex.cloud/en/docs/foundation-models/concepts/yandexgpt/models)
    """

    PRO = "gpt://{folder_id}/yandexgpt/latest"
    LITE = "gpt://{folder_id}/yandexgpt-lite/latest"
    SUMMARIZATION = "gpt://{folder_id}/summarization/latest"
    FINETUNED_DATASPHERE = "ds://{fn_model_id}"


class ModelLimits(BaseModel):
    """Model for the limits of the generative models.

    Attributes
    ----------
    - `MAX_TOKENS_RESPONSE` (`PositiveInt`): Maximum tokens for the response
    - `MAX_TOKENS_TOTAL` (`PositiveInt`): Maximum total tokens
    """

    MAX_TOKENS_RESPONSE: PositiveInt = 2000
    MAX_TOKENS_TOTAL: PositiveInt = 8000


class GenerativeModelLimits(Enum):
    """Enum for the limits of the generative models.

    Attributes
    ----------
    - `PRO` (`ModelLimits`): YandexGPT Pro (v3: 2000 tokens for response, 8000 tokens total)
    - `LITE` (`ModelLimits`): YandexGPT Lite (v2: 2000 tokens for response, 8000 tokens total)
    - `SUMMARIZATION` (`ModelLimits`): Text recap (same as `LITE`) (v2)
    - `FINETUNE_DATASPHERE` (`ModelLimits`): Finetuned models in DataSphere (same as `PRO`) (v3)

    Docs
    ----
    [yandex.cloud/en/docs/foundation-models/concepts/limits](https://yandex.cloud/en/docs/foundation-models/concepts/limits)
    """

    PRO: ModelLimits = ModelLimits(MAX_TOKENS_RESPONSE=2000, MAX_TOKENS_TOTAL=8000)
    LITE: ModelLimits = ModelLimits(MAX_TOKENS_RESPONSE=2000, MAX_TOKENS_TOTAL=8000)
    # SUMMARIZATION: ModelLimits = "GenerativeModelLimits".LITE.value
    SUMMARIZATION: ModelLimits = ModelLimits(MAX_TOKENS_RESPONSE=2000, MAX_TOKENS_TOTAL=8000)
    # FINETUNE_DATASPHERE: ModelLimits = "GenerativeModelLimits".PRO.value
    FINETUNE_DATASPHERE: ModelLimits = ModelLimits(MAX_TOKENS_RESPONSE=2000, MAX_TOKENS_TOTAL=8000)


class ApiEndpoints(StrEnum):
    """Enum for the endpoints of the YandexGPT API.

    Attributes
    ----------
    - `TEXT_GENERATION`: Synchronous text generation (w/ stream support)
    - `TEXT_GENERATION_ASYNC`: Asynchronous text generation
    - `OPERATIONS`: Operation status
    - `TOKENIZE`: Text tokenization
    - `TOKENIZE_COMPLETION`: Completion request tokenization

    Docs
    ----
    [yandex.cloud/en/docs/foundation-models/concepts/api](https://yandex.cloud/en/docs/foundation-models/concepts/api)
    """

    TEXT_GENERATION = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    TEXT_GENERATION_ASYNC = "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync"
    OPERATIONS = "https://llm.api.cloud.yandex.net/operations/{operation_id}"
    TOKENIZE = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenize"
    TOKENIZE_COMPLETION = "https://llm.api.cloud.yandex.net/foundationModels/v1/tokenizeCompletion"
