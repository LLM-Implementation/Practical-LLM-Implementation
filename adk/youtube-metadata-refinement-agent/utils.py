import logging
import json
import re
from typing import Dict, List, Union

logger = logging.getLogger(__name__)

def load_instruction(file_path: str) -> str:
    """Loads instruction text from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Instruction file not found: {file_path}")
        return f"ERROR: Instruction file not found at {file_path}"
    except Exception as e:
        logger.error(f"Error loading instruction file {file_path}: {e}", exc_info=True)
        return f"ERROR: Could not load instruction file {file_path}. Reason: {e}"

def _parse_json_string(json_string: str, context_key: str = "Unknown") -> Union[Dict, List, None]:
    """
    Cleans potential markdown fences and parses a JSON string.
    Returns the parsed object (dict or list) or None if parsing fails.
    Logs warnings on failure.
    """
    if not isinstance(json_string, str):
        logger.warning(f"[_parse_json_string] Input for key '{context_key}' is not a string: {type(json_string)}. Returning None.")
        return None

    # Remove markdown fences (```json ... ``` or ``` ... ```)
    cleaned_string = re.sub(r"^```(?:json)?\s*|\s*```$", "", json_string, flags=re.MULTILINE | re.DOTALL).strip()

    if not cleaned_string:
        logger.warning(f"[_parse_json_string] String for key '{context_key}' is empty after cleaning. Returning None.")
        return None

    try:
        return json.loads(cleaned_string)
    except json.JSONDecodeError as e:
        logger.warning(f"[_parse_json_string] Failed to parse JSON for key '{context_key}'. Error: {e}. String (first 100 chars): '{cleaned_string[:100]}...'")
        return None
    except Exception as e:
        logger.error(f"[_parse_json_string] Unexpected error parsing JSON for key '{context_key}'. Error: {e}. String (first 100 chars): '{cleaned_string[:100]}...'", exc_info=True) # Add stack trace for unexpected errors
        return None