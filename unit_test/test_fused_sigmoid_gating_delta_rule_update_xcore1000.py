"""Accuracy and dispatch isolation for the C500 GLM5.3-flash specialization."""

from dataclasses import replace

import pytest
import torch

from mcoplib import triton_fused_sigmoid_gating_delta_rule_update as implementation
from mcoplib import triton_fused_sigmoid_gating_delta_rule_update_xcore1000 as platform
from unit_test.test_fused_sigmoid_gating_delta_rule_update_target_shapes import (
    CASES,
    assert_correct,
    make_inputs,
)


@pytest.mark.parametrize("arch,expected", (
    ("xcore1000", True), ("xcore1008", True), ("xcore1500", False),
    ("xcore1600", False), (None, False),
))
def test_exact_platform_allowlist(monkeypatch, arch, expected):
    monkeypatch.setattr(platform, "get_xcore_arch", lambda index: arch)
    assert platform.is_supported_xcore(0) is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("arch", ("xcore1000", "xcore1008"))
def test_glm53_flash_platform_dispatch(monkeypatch, arch):
    # The xcore1008 case verifies routing on the available GPU; it is not a
    # claim that the xcore1008 device binary was hardware validated.
    monkeypatch.setattr(platform, "get_xcore_arch", lambda index: arch)
    launched = []
    original_factory = platform.workspace_base_kernels

    class RecordingCandidateKernel:
        def __init__(self, kernel):
            self.kernel = kernel

        def __getitem__(self, grid):
            launched.append(grid)
            return self.kernel[grid]

    def recording_factory(precompute, n1, n2):
        assert n2 is platform.glm53_flash_n2_recurrent_kernel
        gate_kernel, recurrent = original_factory(precompute, n1, n2)
        return gate_kernel, RecordingCandidateKernel(recurrent)

    monkeypatch.setattr(platform, "workspace_base_kernels", recording_factory)
    assert_correct(make_inputs(CASES[0], torch.device("cuda:0"), packed_qkv=True), "auto")
    assert launched == [(128, 4)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
def test_other_platform_preserves_existing_dispatch(monkeypatch):
    monkeypatch.setattr(platform, "get_xcore_arch", lambda index: "xcore1600")
    launched = []
    original = implementation.fused_sigmoid_gating_delta_rule_update_kernel_n2_static6

    class RecordingExistingKernel:
        def __getitem__(self, grid):
            launched.append(grid)
            return original[grid]

    class UnexpectedCandidateKernel:
        def __getitem__(self, grid):
            raise AssertionError("C500 candidate launched on a non-allowlisted platform")

    monkeypatch.setattr(platform, "glm53_flash_n2_recurrent_kernel", UnexpectedCandidateKernel())
    def unexpected_factory(*args):
        raise AssertionError("workspace-base candidate launched on a non-allowlisted platform")
    monkeypatch.setattr(platform, "workspace_base_kernels", unexpected_factory)
    monkeypatch.setattr(implementation, "fused_sigmoid_gating_delta_rule_update_kernel_n2_static6",
                        RecordingExistingKernel())
    assert_correct(make_inputs(CASES[0], torch.device("cuda:0"), packed_qkv=True), "auto")
    assert launched == [(128, 4)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("case,strategy", (
    (CASES[1], "dual_opt_static6_norms_n1"), (CASES[0], "dual_opt_static6_norms_n2"),
))
def test_explicit_static6_strategies_are_preserved(monkeypatch, case, strategy):
    def unexpected_probe(index):
        raise AssertionError("platform dispatch must only run for auto")

    monkeypatch.setattr(platform, "is_supported_xcore", unexpected_probe)
    assert_correct(make_inputs(case, torch.device("cuda:0"), packed_qkv=True), strategy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("arch", ("xcore1000", "xcore1008", "xcore1600"))
def test_n1_keeps_existing_recurrent_kernel(monkeypatch, arch):
    monkeypatch.setattr(platform, "get_xcore_arch", lambda index: arch)
    launched = []
    original = implementation.fused_sigmoid_gating_delta_rule_update_kernel_n1_static6

    class RecordingN1Kernel:
        def __getitem__(self, grid):
            launched.append(grid)
            return original[grid]

    class UnexpectedN2Kernel:
        def __getitem__(self, grid):
            raise AssertionError("N1 must retain its existing recurrent kernel")

    if arch == "xcore1600":
        monkeypatch.setattr(implementation, "fused_sigmoid_gating_delta_rule_update_kernel_n1_static6",
                            RecordingN1Kernel())
        monkeypatch.setattr(platform, "glm53_flash_n2_recurrent_kernel", UnexpectedN2Kernel())
    else:
        original_factory = platform.workspace_base_kernels

        class RecordingBaseKernel:
            def __init__(self, kernel):
                self.kernel = kernel

            def __getitem__(self, grid):
                launched.append(grid)
                return self.kernel[grid]

        def recording_factory(precompute, n1, n2):
            assert n1 is original
            gate_kernel, recurrent = original_factory(precompute, n1, n2)
            return gate_kernel, RecordingBaseKernel(recurrent)

        monkeypatch.setattr(platform, "workspace_base_kernels", recording_factory)
    assert_correct(make_inputs(CASES[1], torch.device("cuda:0"), packed_qkv=True), "auto")
    assert launched == [(64, 8)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
@pytest.mark.parametrize("negative_initial,negative_cache", (
    (True, False), (False, True), (True, True),
))
@pytest.mark.parametrize("strategy", ("original", "auto"))
def test_glm53_flash_negative_slot_indices(negative_initial, negative_cache, strategy):
    spec = replace(
        CASES[0], h0_indices=(-1, 1) if negative_initial else (0, 1),
        intermediate_state_indices=(-1, 3) if negative_cache else (2, 3),
    )
    assert_correct(make_inputs(spec, torch.device("cuda:0"), packed_qkv=True), strategy)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a MACA/CUDA GPU")
def test_glm53_flash_envelope_state_stride():
    inputs = make_inputs(CASES[0], torch.device("cuda:0"), packed_qkv=True)
    envelope = torch.full((7, 2, 64, 128, 128), 1234.5,
                          device=inputs.q.device, dtype=torch.float32)
    envelope[:, 0].copy_(inputs.h0_source)
    inputs.h0_source = envelope[:, 0]
    assert inputs.h0_source.stride(0) == 2 * 1048576
    assert_correct(inputs, "auto")
    assert torch.all(envelope[:, 1] == 1234.5).item()
