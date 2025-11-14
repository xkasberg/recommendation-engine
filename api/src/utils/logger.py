import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional


_CONFIGURED_LOGGERS: set[str] = set()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a configured logger that logs to stdout at INFO by default.

    Respects env var LOG_LEVEL (default INFO) and LOG_FORMAT.
    Idempotent per logger name to avoid duplicate handlers.
    """
    logger_name = name or "virgil"
    logger = logging.getLogger(logger_name)

    if logger_name in _CONFIGURED_LOGGERS:
        return logger

    level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    log_json = os.getenv("LOG_JSON", "1") not in ("0", "false", "False")
    handler = logging.StreamHandler(stream=sys.stdout)
    if log_json:
        # Add the desired level prefix, but keep JSON body clean
        fmt = os.getenv("LOG_FORMAT") or "%(levelname)s:     %(message)s"
    else:
        fmt = os.getenv("LOG_FORMAT") or "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))

    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False

    _CONFIGURED_LOGGERS.add(logger_name)
    return logger


class Logger:
    """Light wrapper that supports structured JSON logs.

    Env:
      - LOG_LEVEL (default INFO)
      - LOG_JSON=1 enables JSON lines
    """

    def __init__(self, name: Optional[str] = None):
        self._name = name or "virgil"
        self._log = get_logger(self._name)
        self._json = os.getenv("LOG_JSON", "1") not in ("0", "false", "False")

    def _emit(self, level: str, msg: str, **kv):
        if self._json:
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": msg,
            }
            if kv:
                payload.update(kv)
            self._log.log(getattr(logging, level.upper(), logging.INFO), json.dumps(payload, ensure_ascii=False))
        else:
            if kv:
                kv_str = " ".join(f"{k}={v}" for k, v in kv.items())
                self._log.log(getattr(logging, level.upper(), logging.INFO), f"{msg} | {kv_str}")
            else:
                self._log.log(getattr(logging, level.upper(), logging.INFO), msg)

    def info(self, msg: str, **kv):
        self._emit("INFO", msg, **kv)

    def warn(self, msg: str, **kv):
        self._emit("WARNING", msg, **kv)

    def error(self, msg: str, **kv):
        self._emit("ERROR", msg, **kv)

    def debug(self, msg: str, **kv):
        self._emit("DEBUG", msg, **kv)
