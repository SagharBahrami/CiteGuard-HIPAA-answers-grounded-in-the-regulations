"""Shared test helpers -- a minimal stand-in for openai.OpenAI covering only
the calls this project actually makes (embeddings.create, chat.completions
.create, chat.completions.parse), so tests never hit the real API.
"""

import types

DEFAULT_USAGE = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)


class FakeOpenAIClient:
    def __init__(self, embedding_vectors=None, chat_content=None, parsed_result=None, usage=None):
        self.calls = []
        self.embeddings = types.SimpleNamespace(create=self._create_embeddings)
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create_chat, parse=self._parse_chat)
        )
        self._embedding_vectors = embedding_vectors
        self._chat_content = chat_content
        self._parsed_result = parsed_result
        self._usage = usage or DEFAULT_USAGE

    def _create_embeddings(self, input, model):
        self.calls.append(("embeddings.create", input, model))
        vectors = self._embedding_vectors(input) if callable(self._embedding_vectors) else self._embedding_vectors
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=v) for v in vectors])

    def _create_chat(self, model, messages):
        self.calls.append(("chat.completions.create", model, messages))
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self._chat_content))],
            usage=self._usage,
        )

    def _parse_chat(self, model, response_format, messages):
        self.calls.append(("chat.completions.parse", model, messages))
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(parsed=self._parsed_result))],
            usage=self._usage,
        )
