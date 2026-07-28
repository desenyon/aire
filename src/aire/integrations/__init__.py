"""First-party provider integrations.

Every integration is optional and lazy: importing :mod:`aire` never imports
these modules. They activate either on first use of a ``provider:name``
reference (via :func:`aire.models.registry._maybe_hint_integration`) or
explicitly through ``aire.integrations.<provider>.register(runtime)``.

All integrations speak to vendors over plain HTTP (httpx) — no vendor SDKs are
required.
"""
