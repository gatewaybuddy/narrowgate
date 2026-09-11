"""narrowgate — an agent harness whose safety argument does not depend on the model's cooperation.

See ``docs/ARCHITECTURE.md``; that file is the contract. This package deliberately exports nothing
at import time beyond its version: every capability is reached by an explicit import of the module
that owns it, never by attribute magic.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
