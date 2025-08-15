import json
import re
import time
from typing import Type, TypeVar

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

# Final channel content extraction regex (non-greedy, stop at end/return)
_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)",
    re.DOTALL,
)
# Fenced JSON blocks
_CODE_FENCE_BLOCK_RE = re.compile(
    r"```(?:json|jsonc|json5)?\s*(.*?)\s*```",
    re.DOTALL | re.IGNORECASE,
)

T = TypeVar("T", bound=BaseModel)


class HarmonyOllamaClient:
    """
    A client for interacting with an Ollama server using the Harmony format.

    This client renders Harmony prompts, calls Ollama /api/generate with raw mode,
    and parses the payload robustly (final channel -> raw JSON -> fenced JSON).
    It also includes retry logic and Pydantic validation for JSON outputs.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gpt-oss:20b",
        timeout: float = 200.0,
        retries: int = 1,
        *,
        # Generation controls to reduce timeouts / verbosity
        num_predict: int = 512,
        num_ctx: int = 8192,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        """
        Args:
            base_url: Ollama base URL.
            model: model name.
            timeout: HTTP timeout (seconds).
            retries: number of retries on failure.
            num_predict: max tokens to generate.
            num_ctx: context window size.
            temperature: sampling temperature.
            top_p: nucleus sampling.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.enc = load_harmony_encoding(HName.HARMONY_GPT_OSS)
        self._gen_options = {
            "num_predict": num_predict,
            "num_ctx": num_ctx,
            "temperature": temperature,
            "top_p": top_p,
        }

    def _render_prompt(self, user_text: str, json_schema_hint: str | None = None) -> str:
        """
        Render a Harmony-formatted prompt.
        We “force” the assistant to start in the final channel to reduce analysis.
        """
        system_message = SystemContent(
            content=(
                "You are a helpful assistant. Respond ONLY in the `final` channel. "
                "Do NOT output the `analysis` channel. Output a valid JSON value (object or array) "
                "and do not include any other text, markdown, or code fences."
            )
        )
        developer_message = DeveloperContent(
            content=(
                (json_schema_hint or "Return a valid JSON object that conforms to the requested schema.")
                + "\nSTRICT REQUIREMENTS:\n"
                "- Respond only in the final channel.\n"
                "- No code fences (no ``` of any kind).\n"
                "- No explanations or extra text outside JSON."
            )
        )
        conversation = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, system_message),
                Message.from_role_and_content(Role.DEVELOPER, developer_message),
                Message.from_role_and_content(Role.USER, user_text),
            ]
        )
        token_ids = self.enc.render_conversation_for_completion(conversation, Role.ASSISTANT)
        prompt = self.enc.decode(token_ids)
        # Pseudo “no-thinking”: start writing directly to final channel
        if not prompt.endswith("<|channel|>final<|message|>"):
            prompt = prompt + "<|channel|>final<|message|>"
        return prompt

    def _call_ollama_api(self, prompt: str, debug: bool = False) -> str:
        """
        Call Ollama /api/generate with raw mode and generation options.
        """
        # Backward-compat guard
        gen_options = getattr(
            self,
            "_gen_options",
            {"num_predict": 512, "num_ctx": 8192, "temperature": 0.2, "top_p": 0.9},
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,
            "stop": ["<|return|>"],
            "stream": False,
            "options": gen_options,
        }
        if debug:
            print(
                f"--- Sending payload to Ollama ---\n{json.dumps(payload, indent=2)}\n---------------------------------",
                flush=True,
            )

        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("response", "")

    def _extract_final_payload(self, text: str) -> str:
        """
        Extract usable payload in order of preference:
        1) Harmony final channel.
        2) Whole response as raw JSON (object/array).
        3) First fenced JSON block (json*/no-language).
        4) Fallback: return stripped text.
        """
        s = (text or "").strip()

        m = _FINAL_RE.search(s)
        if m:
            return m.group(1).strip()

        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            return s

        if "```" in s:
            m2 = _CODE_FENCE_BLOCK_RE.search(s) or re.search(r"```\s*(.*?)\s*```", s, re.DOTALL)
            if m2:
                return m2.group(1).strip()

        return s

    def generate_json(
        self,
        user_text: str,
        pydantic_model: Type[T],
        json_schema_hint: str | None = None,
        debug: bool = False,
    ) -> T:
        """
        Generate JSON from user_text and validate with pydantic_model.
        Retries on request/parse/validation failures.
        """
        # Sanitize user input against special harmony tokens
        special_tokens = ["<|start|>", "<|end|>", "<|channel|>", "<|message|>", "<|return|>"]
        sanitized_user_text = user_text
        for token in special_tokens:
            sanitized_user_text = sanitized_user_text.replace(token, f"[{token.strip('<|>')}]")

        last_exception = None
        raw_response_dump_path = "ollama_error_response.txt"

        for attempt in range(self.retries + 1):
            current_user_text = sanitized_user_text
            if attempt > 0:
                current_user_text += (
                    "\n\n--- IMPORTANT ---\n"
                    "Return ONLY valid JSON with no code fences and no extra text."
                )

            prompt = self._render_prompt(current_user_text, json_schema_hint)

            try:
                raw_response = self._call_ollama_api(prompt, debug=debug)
                if debug:
                    print(
                        f"--- Raw Ollama Response (Attempt {attempt + 1}) ---\n{raw_response}",
                        flush=True,
                    )

                final_text = self._extract_final_payload(raw_response)

                # Keep a strict stripper for the common ```json ... ``` pattern
                m = re.match(
                    r"```(?:json)?\s*\n(.*?)\n```(?:\s*)$",
                    final_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                if m:
                    final_text = m.group(1).strip()

                validated_data = pydantic_model.model_validate_json(final_text)
                return validated_data

            except (requests.RequestException, json.JSONDecodeError, ValidationError) as e:
                last_exception = e
                print(f"Warning: Attempt {attempt + 1} failed. Reason: {e}", flush=True)
                # Save raw response for debugging
                try:
                    with open(raw_response_dump_path, "w") as f:
                        f.write(raw_response if "raw_response" in locals() else "(no response captured)")
                except Exception:
                    pass

                if attempt < self.retries:
                    time.sleep(1)
                else:
                    print(
                        f"Error: Final attempt failed. Raw response saved to {raw_response_dump_path}",
                        flush=True,
                    )

            except RuntimeError as e:
                last_exception = e
                print(f"Warning: Attempt {attempt + 1} failed. Reason: {e}", flush=True)
                if attempt >= self.retries:
                    print("Error: Final attempt failed to find 'final' channel.", flush=True)

        raise RuntimeError(
            f"Failed to get a valid JSON response after {self.retries + 1} attempts."
        ) from last_exception
