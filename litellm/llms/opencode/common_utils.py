from litellm.llms.base_llm.chat.transformation import BaseLLMException


class OpenCodeException(BaseLLMException):
    """Exception for OpenCode API errors."""
