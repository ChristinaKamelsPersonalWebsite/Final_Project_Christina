from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Dict, List


class InMemorySessionStore:
    def __init__(self) -> None:
        self._store: Dict[str, List[dict]] = defaultdict(list)
        self._lock = Lock()

    def get_history(self, session_id: str) -> List[dict]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def append_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            self._store[session_id].append({"role": role, "content": content})

    def clear_history(self, session_id: str) -> None:
        with self._lock:
            self._store.pop(session_id, None)


session_store = InMemorySessionStore()