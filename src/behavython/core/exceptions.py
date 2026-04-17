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
