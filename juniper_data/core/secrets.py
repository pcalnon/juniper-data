"""Docker secrets support for juniper-data."""

import os
from pathlib import Path


def get_secret(env_var: str, file_env_var: str | None = None) -> str | None:
    """Read a secret value, preferring file-based Docker secrets over env vars.

    Args:
        env_var: Environment variable name for the secret value.
        file_env_var: Environment variable name containing a file path to read
            the secret from. If not provided, defaults to ``{env_var}_FILE``.

    Returns:
        The secret value, or None if neither source is set.
    """
    if file_env_var is None:
        file_env_var = f"{env_var}_FILE"
    file_path = os.environ.get(file_env_var)
    if file_path:
        path = Path(file_path)
        if path.is_file():
            return path.read_text().strip()
    return os.environ.get(env_var)
