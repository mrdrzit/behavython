class BehavythonError(Exception):
    """Base exception for all custom Behavython errors."""
    pass

class AnalysisError(BehavythonError):
    """
    Raised when an error breaks the current analysis (e.g., missing bodyparts, 
    invalid arena config, missing video frame, incompatible data), but the 
    program itself is still stable and can continue with other tasks.
    """
    pass

class CriticalSystemError(BehavythonError):
    """
    Raised when an error is program-breaking (e.g., missing critical dependencies, 
    database corruption, read/write permission denied on the root workspace).
    Requires the application to halt.
    """
    pass


# ------------------------------------------------------------------
# DLC Session exceptions
# ------------------------------------------------------------------

class DLCSessionError(BehavythonError):
    """Base exception for all DLC session and assisted-labeling errors."""
    pass

class ProjectIntegrityError(DLCSessionError):
    """
    Raised when the DLC project structure is inconsistent or broken.
    Examples: config.yaml is missing, labeled-data folder not found,
    project folders were moved or renamed after training.
    """
    pass

class ScorerMismatchError(DLCSessionError):
    """
    Raised when the scorer stored in an existing label file does not match
    the scorer in the current config.yaml.
    This must be resolved before any merge can safely proceed.
    """
    pass

class BodypartMismatchError(DLCSessionError):
    """
    Raised when the bodyparts in an existing label file do not match
    the bodyparts defined in the current config.yaml.
    """
    pass

class BackupError(DLCSessionError):
    """
    Raised when the session fails to create a required backup before
    modifying an existing label file. No changes will be made if this is raised.
    """
    pass

class MergeError(DLCSessionError):
    """
    Raised when the H5/CSV merge step fails after inference.
    The session will attempt a rollback using the pre-merge backups.
    """
    pass
