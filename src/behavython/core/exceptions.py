class BehavythonError(Exception):
    """Base exception for all custom Behavython errors."""


class BackendError(BehavythonError):
    """Base exception for DLC backend errors."""


class UnsupportedBackendError(BackendError):
    """Raised when a DLC config specifies an unsupported engine."""


class MissingBackendError(BackendError):
    """Raised when the backend required by the DLC network is not installed."""


class AnalysisError(BehavythonError):
    """
    Raised when an error breaks the current analysis (e.g., missing bodyparts,
    invalid arena config, missing video frame, incompatible data), but the
    program itself is still stable and can continue with other tasks.
    """


class CriticalSystemError(BehavythonError):
    """
    Raised when an error is program-breaking (e.g., missing critical dependencies,
    database corruption, read/write permission denied on the root workspace).
    Requires the application to halt.
    """


# ------------------------------------------------------------------
# DLC Session exceptions
# ------------------------------------------------------------------


class DLCSessionError(BehavythonError):
    """Base exception for all DLC session and assisted-labeling errors."""


class ProjectIntegrityError(DLCSessionError):
    """
    Raised when the DLC project structure is inconsistent or broken.
    Examples: config.yaml is missing, labeled-data folder not found,
    project folders were moved or renamed after training.
    """


class ScorerMismatchError(DLCSessionError):
    """
    Raised when the scorer stored in an existing label file does not match
    the scorer in the current config.yaml.
    This must be resolved before any merge can safely proceed.
    """


class BodypartMismatchError(DLCSessionError):
    """
    Raised when the bodyparts in an existing label file do not match
    the bodyparts defined in the current config.yaml.
    """


class BackupError(DLCSessionError):
    """
    Raised when the session fails to create a required backup before
    modifying an existing label file. No changes will be made if this is raised.
    """


class MergeError(DLCSessionError):
    """
    Raised when the H5/CSV merge step fails after inference.
    The session will attempt a rollback using the pre-merge backups.
    """
