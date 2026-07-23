# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for AMP runner checkpoint loading."""

from __future__ import annotations

import torch
from pathlib import Path
from unittest.mock import Mock

from rsl_rl.runners import AMPOnPolicyRunner


def test_load_optimizer_choice_is_forwarded_to_algorithm(tmp_path: Path) -> None:
    """The runner should pass both optimizer-loading choices to the algorithm."""
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"iter": 7, "infos": {"source": "test"}}, checkpoint_path)

    runner = AMPOnPolicyRunner.__new__(AMPOnPolicyRunner)
    runner.alg = Mock()
    runner.alg.load.return_value = True
    runner.current_learning_iteration = 0

    for load_optimizer in (False, True):
        infos = runner.load(str(checkpoint_path), load_optimizer=load_optimizer)

        load_cfg = runner.alg.load.call_args.kwargs["load_cfg"]
        assert load_cfg["optimizer"] is load_optimizer
        assert runner.current_learning_iteration == 7
        assert infos == {"source": "test"}
