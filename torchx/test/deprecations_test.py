# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for :py:mod:`torchx.deprecations`."""

import unittest
import warnings

from torchx.deprecations import _PLUGIN_GROUPS, deprecated_entrypoint


class DeprecatedEntrypointTest(unittest.TestCase):
    """Tests for ``deprecated_entrypoint()``."""

    def test_warns_for_scheduler_group(self) -> None:
        """Emits DeprecationWarning for the 'torchx.schedulers' group."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.schedulers", ["local_cwd", "slurm"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            1,
            "should emit exactly one DeprecationWarning for torchx.schedulers",
        )
        msg = str(dep_warnings[0].message)
        self.assertIn(
            "torchx.schedulers",
            msg,
            "warning should mention the deprecated group",
        )
        self.assertIn(
            "torchx_plugins.schedulers",
            msg,
            "warning should mention the namespace-package alternative",
        )
        self.assertIn(
            "TORCHX_NO_ENTRYPOINTS=1",
            msg,
            "warning should mention the opt-out env var",
        )

    def test_warns_for_named_resource_group(self) -> None:
        """Emits DeprecationWarning for the 'torchx.named_resources' group."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.named_resources", ["aws_p5"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            1,
            "should emit exactly one DeprecationWarning for torchx.named_resources",
        )

    def test_warns_for_tracker_group(self) -> None:
        """Emits DeprecationWarning for the 'torchx.tracker' group."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.tracker", ["mlflow"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            1,
            "should emit exactly one DeprecationWarning for torchx.tracker",
        )

    def test_no_warning_for_orchestrator_group(self) -> None:
        """No warning for 'torchx.schedulers.orchestrator' — no namespace alternative."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.schedulers.orchestrator", ["fblearner"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            0,
            "should not warn for groups without namespace-plugin alternatives",
        )

    def test_no_warning_for_components_group(self) -> None:
        """No warning for 'torchx.components' — no namespace alternative."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.components", ["my_component"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            0,
            "should not warn for torchx.components group",
        )

    def test_no_warning_for_file_group(self) -> None:
        """No warning for 'torchx.file' — no namespace alternative."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.file", ["get_file_contents"])

        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        self.assertEqual(
            len(dep_warnings),
            0,
            "should not warn for torchx.file group",
        )

    def test_warning_includes_sorted_plugin_names(self) -> None:
        """Warning message lists plugin names in sorted order."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deprecated_entrypoint("torchx.schedulers", ["zz_scheduler", "aa_scheduler"])

        msg = str(caught[0].message)
        self.assertIn(
            "aa_scheduler, zz_scheduler",
            msg,
            "plugin names should be sorted alphabetically in the warning",
        )

    def test_plugin_groups_matches_plugin_type(self) -> None:
        """_PLUGIN_GROUPS covers all PluginType values."""
        from torchx.plugins._registry import PluginType

        expected = {pt.value for pt in PluginType}
        self.assertEqual(
            _PLUGIN_GROUPS,
            expected,
            "_PLUGIN_GROUPS should match PluginType values exactly. "
            "If you added a new PluginType, add it to _PLUGIN_GROUPS too.",
        )
