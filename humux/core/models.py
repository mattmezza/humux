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
class OutputMessage:
    """One delivered message in a turn (#202).

    A turn may send several of these in order — multiple text bubbles, or a mix
    of text and voice. ``text`` is always the words (kept even for a voice
    message, so history records what was said); ``voice`` carries synthesized
    audio when the segment was voice-marked.
    """

    text: str = ""
    voice: bytes | None = None


@dataclass
class AgentResponse:
    text: str
    voice: bytes | None = None
    attachments: list[Attachment] = field(default_factory=list)
    # Optional out-of-band system message (e.g. "context was compacted"),
    # delivered by the channel as a separate follow-up message.
    system_notice: str | None = None
    # Ordered messages to deliver this turn (#202). Empty for turns built the
    # old way (control commands, scheduler); ``delivery_messages`` falls back to
    # the flat text/voice fields so every existing caller keeps working.
    messages: list[OutputMessage] = field(default_factory=list)

    @property
    def delivery_messages(self) -> list[OutputMessage]:
        """The messages a channel should send, in order.

        Prefers the explicit ``messages`` list; otherwise derives a single
        message from the flat ``text``/``voice`` fields (backward compatible).
        """
        if self.messages:
            return self.messages
        if self.text or self.voice:
            return [OutputMessage(text=self.text, voice=self.voice)]
        return []
