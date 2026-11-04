"""Load secrets into the environment: .env locally, Colab Secrets in Colab.

Secrets are only read into os.environ. They are never printed, logged or written to disk.
"""

import os

from src.config import REPO_ROOT

SECRET_NAMES = ["HF_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY", "WANDB_API_KEY"]


def load_secrets(names=SECRET_NAMES):
    """Fill missing secrets from .env (python-dotenv) or Colab Secrets. Returns the names found."""
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:
        pass  # python-dotenv is optional (for example on Kaggle)

    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        try:
            from google.colab import userdata  # only inside a Colab kernel
        except ImportError:
            userdata = None
        for name in missing if userdata else []:
            try:
                value = userdata.get(name)
            except Exception:  # secret not defined or notebook access not granted
                continue
            if value:
                os.environ[name] = value
    return [name for name in names if os.environ.get(name)]
