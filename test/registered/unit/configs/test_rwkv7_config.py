"""CPU tests for the RWKV-7 SGLang configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.overrides import (
    ResolvedView,
    _mamba_radix_cache_resolution,
)
from sglang.srt.configs.linear_attn_model_registry import (
    get_linear_attn_config,
    get_linear_attn_spec_by_arch,
)
from sglang.srt.configs.rwkv7 import Rwkv7Config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRwkv7Config(CustomTestCase):
    def test_linear_attention_registration(self):
        config = Rwkv7Config()
        result = get_linear_attn_config(config)

        self.assertIsNotNone(result)
        spec, resolved_config = result
        self.assertIs(resolved_config, config)
        self.assertTrue(spec.uses_mamba_radix_cache)
        self.assertTrue(spec.support_mamba_cache)
        self.assertFalse(spec.support_mamba_cache_extra_buffer)
        self.assertIs(
            get_linear_attn_spec_by_arch("RWKV7ForCausalLM"),
            spec,
        )
        self.assertIs(
            get_linear_attn_spec_by_arch("Rwkv7ForCausalLM"),
            spec,
        )

    def test_auto_selects_no_buffer_state_cache(self):
        hf_config = SimpleNamespace(architectures=["RWKV7ForCausalLM"])
        server_args = SimpleNamespace(
            get_model_config=lambda: SimpleNamespace(hf_config=hf_config),
            disable_radix_cache=False,
            mamba_radix_cache_strategy="auto",
            disable_overlap_schedule=False,
            page_size=None,
            linear_attn_backend="triton",
        )

        self.assertEqual(
            _mamba_radix_cache_resolution(ResolvedView(server_args)),
            {
                "uses_mamba_radix_cache": True,
                "mamba_radix_cache_strategy": "no_buffer",
                "disable_overlap_schedule": True,
            },
        )

    def test_explicit_radix_disable_is_respected(self):
        hf_config = SimpleNamespace(architectures=["RWKV7ForCausalLM"])
        server_args = SimpleNamespace(
            get_model_config=lambda: SimpleNamespace(hf_config=hf_config),
            disable_radix_cache=True,
            mamba_radix_cache_strategy="auto",
            disable_overlap_schedule=False,
            page_size=None,
            linear_attn_backend="triton",
        )

        self.assertEqual(
            _mamba_radix_cache_resolution(ResolvedView(server_args)),
            {},
        )


if __name__ == "__main__":
    unittest.main()
