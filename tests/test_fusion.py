# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test operator fusion."""

# pylint: disable=comparison-with-callable
import copy
import time
import operator
import pytest

import torch
from torch import nn
import torch.nn.functional as F

import slapo
from slapo.pattern import call_module


def test_decompose():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 20)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.linear(x)
            x = self.relu(x)
            return x

    mod = Model().cuda()
    sch = slapo.create_schedule(copy.deepcopy(mod))

    sch["linear"].decompose()
    sch.trace(flatten=True)

    sch_model, _ = slapo.build(sch, init_weights=False)
    print(sch_model)

    inp = torch.randn((32, 10), requires_grad=True).cuda()
    out = sch_model(inp)
    out_ref = mod(inp)
    torch.testing.assert_close(out, out_ref)


def test_vertical_fusion():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3)

        def forward(self, x):
            x = self.conv(x)
            x = F.relu(x)
            x = x + 1
            return x

    mod = Model().cuda()
    sch = slapo.create_schedule(copy.deepcopy(mod))

    def pattern(x: torch.Tensor):
        x = call_module("conv", x)
        x = F.relu(x)
        x = x + 1
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 3
    sch.fuse(subgraph, compiler="TorchScript", name="FusedReLU")

    sch_model, _ = slapo.build(sch, init_weights=False)
    print(sch_model)

    inp = torch.randn((1, 3, 32, 32), requires_grad=True).cuda()
    out = sch_model(inp)
    out_ref = mod(inp)
    torch.testing.assert_close(out, out_ref)


def test_bias_gelu():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1024, 1024)
            self.gelu = nn.GELU()

        def forward(self, x):
            x = self.linear(x)
            x = self.gelu(x)
            return x

    mod = Model().cuda()
    sch = slapo.create_schedule(copy.deepcopy(mod))

    sch["linear"].decompose()
    sch.trace(flatten=True)
    print(sch.mod.graph)

    def pattern(x, bias):
        x = F.gelu(bias + x)
        return x

    subgraph = sch.find(pattern)
    assert len(subgraph[0]) == 2
    assert subgraph[0][0][1].target == operator.add
    assert subgraph[0][1][1].target == "gelu"
    sch.fuse(subgraph, compiler="TorchScript", name="BiasGeLU")
    assert isinstance(sch["BiasGeLU_0"].mod, torch.jit.ScriptModule)

    sch_model, _ = slapo.build(sch, init_weights=False)
    print(sch_model)

    inp = torch.randn((1, 16, 1024, 1024), requires_grad=True).cuda()
    # Wram up
    for _ in range(1000):
        out = sch_model(inp)
    print("Finish warm-up steps")
    start_time = time.time()
    for _ in range(1000):
        out = sch_model(inp)
    ts_time = time.time() - start_time
    print(f"TorchScript time: {ts_time:.4f}s")
    start_time = time.time()
    for _ in range(1000):
        out_ref = mod(inp)
    vanilla_time = time.time() - start_time
    print(f"Vanilla time: {vanilla_time:.4f}s")
    torch.testing.assert_close(out, out_ref)
    # Performance testing
    assert ts_time < vanilla_time


if __name__ == "__main__":
    pytest.main([__file__])