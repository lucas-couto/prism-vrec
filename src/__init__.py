"""prism-vrec — a reproducible framework for evaluating visual feature
extractors in recommender systems.

``__version__`` is the SINGLE source of truth for the framework version.
``pyproject.toml`` reads it via ``[tool.setuptools.dynamic]`` rather than
declaring its own, so the packaged metadata and the running code can
never disagree.

This matters for the run manifest.  ``importlib.metadata`` reads the
``dist-info`` written at *install* time; under the Docker setup that
bind-mounts ``./src`` over an editable install, that metadata is frozen
at whatever version the image was built with, while the code executing
is whatever the mount provides.  Reading the version from this module
instead makes the manifest report the code that actually ran.
"""

__version__ = "2.9.0"
