
class StateTrackerError(Exception):
    """Base exception for state tracker module."""


class InvalidTrackedStateSpecError(StateTrackerError):
    """Raised when tracked state definitions are invalid."""


class StateProviderContractError(StateTrackerError):
    """Raised when provider output violates expected contract."""


class MissingStateValueError(StateTrackerError):
    """Raised when a required state is not returned by provider."""


class StateTypeCoercionError(StateTrackerError):
    """Raised when a state value cannot be coerced to expected type."""


class StoreContractError(StateTrackerError):
    """Raised when store operations fail."""