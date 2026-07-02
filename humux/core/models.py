"""Shared data models."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field

# Mime types we accept as images for LLM vision.
IMAGE_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
    }
)


@dataclass
class Attachment:
    """A file attachment (image, document, etc.) sent by the user."""

    data: bytes
    mime_type: str
    filename: str | None = None

    @property
    def is_image(self) -> bool:
        return self.mime_type in IMAGE_MIME_TYPES

    @property
    def base64_data(self) -> str:
        return base64.standard_b64encode(self.data).decode("ascii")

    def to_anthropic_block(self) -> dict:
        """Build an Anthropic image content block."""
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": self.mime_type,
                "data": self.base64_data,
            },
        }

    def to_openai_block(self) -> dict:
        """Build an OpenAI-compatible image_url content block."""
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{self.mime_type};base64,{self.base64_data}",
            },
        }


@dataclass
class AgentResponse:
    text: str
    voice: bytes | None = None
    attachments: list[Attachment] = field(default_factory=list)
    # Optional out-of-band system message (e.g. "context was compacted"),
    # delivered by the channel as a separate follow-up message.
    system_notice: str | None = None
