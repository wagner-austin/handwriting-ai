from __future__ import annotations

import contextvars

# Request-scoped correlation id, blank if not set
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
