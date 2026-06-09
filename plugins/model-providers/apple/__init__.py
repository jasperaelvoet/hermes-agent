"""Apple Foundation Models provider profile.

Apple's `fm serve` (macOS 27+) exposes an OpenAI-compatible Chat Completions
server for the on-device ``system`` model and the Private Cloud Compute
``pcc`` model. Hermes drives it through a custom client facade
(:class:`agent.apple_fm_client.AppleFMClient`) managing a single ``fm serve``
subprocess — NOT a remote HTTP endpoint — so this profile only carries
auth/endpoint metadata and routes through the ``chat_completions`` transport
(like the copilot-acp profile). No API key: availability is gated by the
machine's Apple Intelligence entitlements.
"""

from providers import register_provider
from providers.base import ProviderProfile


class AppleFMProfile(ProviderProfile):
    """Apple Foundation Models — external `fm serve` process, fixed catalog."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """No REST catalog — models are the fixed on-device/PCC pair."""
        return None


apple = AppleFMProfile(
    name="apple",
    aliases=("apple-fm", "apple-foundation-models", "foundation-models"),
    api_mode="chat_completions",  # AppleFMClient returns OpenAI-shaped objects
    display_name="Apple Foundation Models",
    description="Apple on-device & Private Cloud Compute models via `fm serve` (macOS 27+) — experimental: context below Hermes' 64K minimum",
    env_vars=(),  # managed external process; no API key
    base_url="applefm://local",  # marker scheme handled in create_openai_client
    auth_type="external_process",
    supports_health_check=False,  # doctor must skip the /models REST probe
    supports_vision=False,
    fallback_models=("apple/system", "apple/pcc"),
)

register_provider(apple)
