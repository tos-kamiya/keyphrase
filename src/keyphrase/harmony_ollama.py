

import json
import re
import time
from typing import Any, Dict, Type, TypeVar

import requests
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName as HName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)
from pydantic import BaseModel, ValidationError

# Final channel content extraction regex
_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*)", re.DOTALL)
T = TypeVar("T", bound=BaseModel)


class HarmonyOllamaClient:
    """
    A client for interacting with an Ollama server using the Harmony format.

    This client handles rendering prompts in the Harmony format, sending them to
    Ollama's `/api/generate` endpoint with `raw: true`, and parsing the
    `final` channel from the response. It also includes retry logic and
    Pydantic validation for JSON outputs.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b",
        timeout: float = 120.0,
        retries: int = 1,
    ):
        """
        Initialize the client.

        Args:
            base_url: The base URL of the Ollama server.
            model: The model name to use for generation.
            timeout: The request timeout in seconds.
            retries: The number of times to retry on failure.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.enc = load_harmony_encoding(HName.HARMONY_GPT_OSS)

    def _render_prompt(self, user_text: str, json_schema_hint: str | None = None) -> str:
        """
        Renders a conversation into a single prompt string using the Harmony format.
        """
        system_message = SystemContent(
            content="You are a helpful assistant. Output a valid JSON object in the final channel. Do not include any other text or explanations."
        )
        developer_message = DeveloperContent(
            content=json_schema_hint or "Return a valid JSON object that conforms to the requested schema."
        )
        conversation = Conversation.from_messages([
            Message.from_role_and_content(Role.SYSTEM, system_message),
            Message.from_role_and_content(Role.DEVELOPER, developer_message),
            Message.from_role_and_content(Role.USER, user_text),
        ])
        token_ids = self.enc.render_conversation_for_completion(conversation, Role.ASSISTANT)
        return self.enc.decode(token_ids)

    def _call_ollama_api(self, prompt: str) -> str:
        """
        Makes a POST request to the Ollama /api/generate endpoint.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stop": ["<|return|>"],
            "stream": False,
        }
        # Add debug print to show the payload being sent
        print(f"--- Sending payload to Ollama ---\n{json.dumps(payload, indent=2)}\n---------------------------------", flush=True)
        
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("response", "")

    def _extract_final_channel(self, text: str) -> str:
        """
        Extracts the content of the 'final' channel from the Harmony response.
        """
        match = _FINAL_RE.search(text)
        if not match:
            raise RuntimeError(f"Harmony 'final' channel not found in response: {text}")
        return match.group(1).strip()

    def generate_json(
        self,
        user_text: str,
        pydantic_model: Type[T],
        json_schema_hint: str | None = None,
        debug: bool = False,
    ) -> T:
        """
        Generates a JSON object from a user prompt and validates it against a Pydantic model.

        This method handles prompt rendering, API calls, response parsing, and validation,
        including a retry mechanism for validation failures.

        Args:
            user_text: The user's input prompt.
            pydantic_model: The Pydantic model to validate the JSON output against.
            json_schema_hint: A hint to the model about the expected JSON schema.
            debug: If True, prints raw responses for debugging.

        Returns:
            A validated Pydantic model instance.

        Raises:
            RuntimeError: If the request fails after all retries or if parsing fails.
        """
        last_exception = None
        for attempt in range(self.retries + 1):
            # On retry, provide a more forceful instruction.
            current_user_text = user_text
            if attempt > 0:
                current_user_text += "\n\n--- IMPORTANT ---\nRespond with a valid JSON object ONLY. Do not add any commentary or extra text outside the JSON structure."

            prompt = self._render_prompt(current_user_text, json_schema_hint)

            try:
                raw_response = self._call_ollama_api(prompt)
                print(f"--- Raw Ollama Response (Attempt {attempt + 1}) ---\n{raw_response}\n---------------------------------", flush=True)

                if debug:
                    print(f"--- Raw Ollama Response (Attempt {attempt + 1}) ---\n{raw_response}", flush=True)

                final_text = self._extract_final_channel(raw_response)
                validated_data = pydantic_model.model_validate_json(final_text)
                return validated_data

            except (requests.RequestException, json.JSONDecodeError, ValidationError) as e:
                last_exception = e
                print(f"Warning: Attempt {attempt + 1} failed. Reason: {e}", flush=True)
                if attempt < self.retries:
                    time.sleep(1)  # Wait a moment before retrying
                else:
                    # Save the actual error response from the server for debugging
                    error_dump_path = "ollama_error_response.txt"
                    error_content = "Response not available."
                    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                        error_content = e.response.text
                    
                    with open(error_dump_path, "w") as f:
                        f.write(error_content)
                    print(f"Error: Final attempt failed. Detailed error response saved to {error_dump_path}", flush=True)

            except RuntimeError as e:
                last_exception = e
                print(f"Warning: Attempt {attempt + 1} failed. Reason: {e}", flush=True)
                if attempt >= self.retries:
                    print(f"Error: Final attempt failed to find 'final' channel.", flush=True)

        raise RuntimeError(f"Failed to get a valid JSON response after {self.retries + 1} attempts.") from last_exception
