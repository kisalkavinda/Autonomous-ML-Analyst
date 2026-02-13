class AnalysisError(Exception):
    """Base class for exceptions in this module."""
    pass

class InsufficientDataError(AnalysisError):
    """Exception raised when there is not enough data to proceed."""
    pass

class TargetConstantError(AnalysisError):
    """Exception raised when the target variable is constant."""
    pass

class ExcessiveMissingDataError(AnalysisError):
    """Exception raised when there is too much missing data."""
    pass
