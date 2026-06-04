"""Custom exceptions for AI-Aquatica-RS."""


class AIAquaticaError(Exception):
    """Base exception for all package-specific errors."""


class ConfigError(AIAquaticaError):
    """Raised when configuration loading or validation fails."""


class DataValidationError(AIAquaticaError):
    """Raised when input data does not match expected schema constraints."""


class CLIError(AIAquaticaError):
    """Raised for command-line invocation and execution errors."""
