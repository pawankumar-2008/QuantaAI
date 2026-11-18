class ProviderError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        provider: str,
        category: str,
        status_code: int | None = None,
        diagnostic: str = "",
    ):
        super().__init__(message)
        self.provider = provider
        self.category = category
        self.status_code = status_code
        self.diagnostic = diagnostic


class ProviderUnavailableError(ProviderError):
    def __init__(self, message: str):
        super().__init__(
            message,
            provider="online",
            category="unavailable",
        )