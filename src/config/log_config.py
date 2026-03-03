import os
from src.config.paths import PATH

log_config = {
    "version": 1,
    "handlers": {
        "app_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "formatter",
            "filename": PATH["app_log"],
            "mode": "a",
            "maxBytes": 10000000,
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "error_handler": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "formatter",
            "filename": PATH["error_log"],
            "mode": "a",
            "maxBytes": 5000000,
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "console_handler": {
            "class": "logging.StreamHandler",
            "formatter": "formatter",
            "level": "INFO",
        }
    },
    "loggers": {
        "app": {
            "handlers": ["app_handler", "console_handler"],
            "level": "INFO",
        },
        "ui": {
            "handlers": ["app_handler", "console_handler"],
            "level": "INFO",
        },
        "camera": {
            "handlers": ["app_handler", "console_handler"],
            "level": "INFO",
        },
        "analysis": {
            "handlers": ["app_handler", "error_handler"],
            "level": "INFO",
        },
        "visualizer": {
            "handlers": ["app_handler"],
            "level": "INFO",
        },
        "data": {
            "handlers": ["app_handler", "error_handler"],
            "level": "INFO",
        },
    },
    "formatters": {
        "formatter": {
            "format": ("%(asctime)s - %(levelname)s - %(name)s - %(funcName)s - %(message)s")
        }
    },
}