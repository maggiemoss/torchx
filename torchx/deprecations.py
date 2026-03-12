# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Deprecation utilities for TorchX.

Shared helpers for emitting structured deprecation warnings across the
TorchX codebase.  Each function targets a specific deprecation pattern
(e.g., entry-point plugins, import stubs) with clear migration guidance.
"""

import warnings
from typing import Iterable

# Entry-point groups that have ``torchx_plugins.*`` namespace-package
# alternatives.  Only these groups trigger a deprecation warning.
# Keep in sync with ``torchx.plugins._registry.PluginType``.
_PLUGIN_GROUPS: frozenset[str] = frozenset(
    {
        "torchx.schedulers",
        "torchx.named_resources",
        "torchx.tracker",
    }
)


def deprecated_entrypoint(
    group: str,
    ep_names: Iterable[str],
    *,
    stacklevel: int = 2,
) -> None:
    """Emit a deprecation warning for entry-point based plugins.

    Only warns for groups that have ``torchx_plugins.*`` namespace-package
    equivalents (i.e., groups listed in
    :py:class:`~torchx.plugins.PluginType`).  Groups without namespace
    alternatives (e.g., ``"torchx.schedulers.orchestrator"``,
    ``"torchx.components"``) are silently ignored.

    Args:
        group: The entry-point group name (e.g., ``"torchx.schedulers"``).
        ep_names: Names of the entry-point plugins that were loaded.
        stacklevel: Stack level for :py:func:`warnings.warn`.  Default ``2``
            points at the caller of this function.

    Example::

        >>> # In _registry._find():
        >>> deprecated_entrypoint("torchx.schedulers", ["mast_conda"])

    """
    if group not in _PLUGIN_GROUPS:
        return

    names = ", ".join(sorted(ep_names))
    namespace = f"torchx_plugins.{group.removeprefix('torchx.')}"
    warnings.warn(
        f"Entry-point plugins in group '{group}' are deprecated. "
        f"Migrate to the '{namespace}' namespace package using "
        f"the @register decorator. "
        f"Set TORCHX_NO_ENTRYPOINTS=1 to opt out early. "
        f"Deprecated entry-point plugins: {names}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
