"""Shared test helpers -- minimal stand-ins for the external services this
project talks to: openai.OpenAI (covering only embeddings.create,
chat.completions.create and chat.completions.parse) and redis.Redis (covering
only .lock()), so tests never hit the real API or need a Redis running.
"""

import types

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import LockNotOwnedError

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


class FakeRedisLock:
    """Mimics redis.lock.Lock's non-blocking acquire/release ownership rules."""

    def __init__(self, store, name, unreachable=False, expire_on_release=False):
        self._store = store
        self._name = name
        self._unreachable = unreachable
        self._expire_on_release = expire_on_release

    def acquire(self, blocking=None, blocking_timeout=None, token=None):
        if self._unreachable:
            raise RedisConnectionError("connection refused")
        if self._name in self._store:
            return False
        self._store[self._name] = self
        return True

    def release(self):
        if self._expire_on_release:
            self._store.pop(self._name, None)
            raise LockNotOwnedError("lock expired before release")
        if self._store.get(self._name) is not self:
            raise LockNotOwnedError("not owned")
        del self._store[self._name]


class FakeRedis:
    """Minimal stand-in for redis.Redis covering only .lock()."""

    def __init__(self, held=(), unreachable=False, expire_on_release=False):
        self.store = {name: object() for name in held}
        self._unreachable = unreachable
        self._expire_on_release = expire_on_release

    def lock(self, name, timeout=None, **kwargs):
        return FakeRedisLock(
            self.store, name,
            unreachable=self._unreachable,
            expire_on_release=self._expire_on_release,
        )
