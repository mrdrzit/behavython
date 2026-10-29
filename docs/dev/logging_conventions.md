# Logging Conventions

How behavython reports what it does. One channel: Python `logging`. Everything else (GUI text boxes, terminal, log file) is a handler on that channel.

## 1. Loggers
- Every module creates its logger once at the top: `logger = logging.getLogger(__name__)`. Never hand-pick a logger name.
- Exception: `cli.py` uses `logging.getLogger("behavython.cli")`, because under `python -m behavython.cli` `__name__` is `"__main__"`, which is outside the `behavython` tree.
- Only `services/logging.py` (`AppLoggingService`) adds handlers, and only to the `behavython` root logger. Module loggers have no handlers and propagate to it.
- `behavython.external` receives captured DeepLabCut and TensorFlow output (`capture_external_output`).
- `behavython.popup` receives popup text through `log_popup` (`services/logging.py`), used by `show_warning` and the CLI warning adapter.

## 2. Where Messages Go
- Session file: `~/.behavython/logs/<YYYY-MM-DD_HH-MM-SS>.log`, one per GUI or CLI session, all levels. The newest `SESSION_LOGS_TO_KEEP` files are kept. `BEHAVYTHON_HOME` moves the whole folder (the regression test uses this).
- Terminal: INFO and higher (DEBUG with `behavython-cli --verbose`). The GUI terminal leaves out `behavython.external`.
- GUI text boxes: INFO and higher from the modules listed in `GUI_LOG_PANELS` (`core/defaults.py`), routed by logger name. Add a module there to show its messages in a box. Warnings and errors get a `Warning:` or `Error:` prefix; `logger.exception` adds the exception text.
- Popups: `warning.emit(title, text)` in task functions, `show_warning` in GUI code. Both write one WARNING line through `log_popup` (a generic "Warning" title is dropped, line breaks become spaces), so do not also log the same event.
- Per-animal issues: `animal.log(level, message, context)` with level `"info"`, `"warning"` or `"error"` and a context such as `"position_loading"`. It writes `analysis_log.json` in the output folder and the session log (prefixed with the animal id). Never append to `animal.logs` directly.

## 3. Message Style
- f-strings: `logger.info(f"Moved {count} file(s) to {folder}")`.
- The level carries the severity. No `[ERROR]`, `WARNING` or `Error:` in the text.
- One plain sentence or a `key=value` list. No trailing `...`, no leading newline or indent, no emoji, arrows or box-drawing characters, no aligned tables.
- Name the subject: file name, animal id, or video name.
- No multi-line listings with one log call per item; join them into one message.

## 4. Levels
- DEBUG: per-file and per-frame detail, internal calls, tracebacks that a popup already reported (`logger.debug("...", exc_info=True)`).
- INFO: a step starts or finishes, settings used by a run, counts.
- WARNING: something was skipped or replaced and the run continues.
- ERROR: an operation failed. Inside `except`, use `logger.exception("...")` so the traceback reaches the file; do not repeat `{e}` in the message (ruff `TRY401`).

## 5. Task Functions
- Signature: `fn(request, progress=None, warning=None)`. There is no `log` parameter; log with the module logger.
- Run from the GUI through `TaskWorker`, from the CLI directly. Both see the same logger output.
