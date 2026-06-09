"""Registration wiring for the Apple Foundation Models provider."""

from __future__ import annotations


def test_provider_profile_registered_with_aliases():
    from providers import get_provider_profile

    p = get_provider_profile("apple")
    assert p is not None
    assert p.name == "apple"
    assert p.api_mode == "chat_completions"
    assert p.auth_type == "external_process"
    assert p.base_url == "applefm://local"
    assert p.supports_health_check is False
    for alias in ("apple-fm", "apple-foundation-models", "foundation-models"):
        assert get_provider_profile(alias) is p


def test_determine_api_mode_is_chat_completions():
    from hermes_cli.providers import determine_api_mode

    assert determine_api_mode("apple") == "chat_completions"


def test_provider_models_catalog():
    from hermes_cli.models import _PROVIDER_MODELS

    assert _PROVIDER_MODELS["apple"] == ["apple/system", "apple/pcc"]


def test_provider_model_ids_returns_static_list():
    from hermes_cli.models import provider_model_ids

    ids = provider_model_ids("apple")
    assert "apple/system" in ids
    assert "apple/pcc" in ids


def test_canonical_provider_row_present():
    from hermes_cli.models import CANONICAL_PROVIDERS

    assert any(p.slug == "apple" for p in CANONICAL_PROVIDERS)


def test_context_lengths_declared():
    from agent.model_metadata import DEFAULT_CONTEXT_LENGTHS as D

    assert D["apple/system"] == 8192
    assert D["apple/pcc"] == 32768


def test_pricing_is_zero_cost():
    from agent.usage_pricing import _OFFICIAL_DOCS_PRICING

    for model in ("apple/system", "apple/pcc"):
        entry = _OFFICIAL_DOCS_PRICING[("apple", model)]
        assert entry.input_cost_per_million == 0
        assert entry.output_cost_per_million == 0


def test_apple_registered_in_auth_registry():
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY["apple"]
    assert cfg.auth_type == "external_process"
    assert cfg.inference_base_url == "applefm://local"
