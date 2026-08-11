"""Project exception hierarchy with actionable failure categories."""


class FallGuardError(Exception):
    """Base exception for expected application failures."""


class ConfigurationError(FallGuardError):
    """Configuration is missing, unknown, or internally inconsistent."""


class UnsupportedConfigurationError(ConfigurationError):
    """The selected third-party implementation does not expose a requested feature."""


class DependencyUnavailableError(FallGuardError):
    """An explicitly requested optional dependency is not installed or loadable."""


class ModelUnavailableError(FallGuardError):
    """A model checkpoint or local model is unavailable."""


class ProviderUnavailableError(FallGuardError):
    """A semantic provider cannot be used in the current environment."""


class PrivacyConsentRequiredError(FallGuardError):
    """A cloud image operation was requested without explicit consent."""


class FormalBenchmarkRejectedError(FallGuardError):
    """A formal benchmark violates evidence or configuration gates."""


class EvaluationProtocolError(FallGuardError):
    """Ground truth or metric protocol is insufficient for a requested metric."""


class NonMonotonicTimestampError(FallGuardError):
    """Frame timestamps did not increase monotonically."""
