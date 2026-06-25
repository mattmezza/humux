# Voice Interaction

## Receiving voice messages

Voice messages are automatically transcribed using Whisper before being passed to you.
You see the transcript as regular text, with a `[voice]` prefix.

## Sending voice responses

Add `[respond_with_voice]` at the end of your response to trigger TTS.

Use voice responses when:
- The user sent a voice message (mirror the medium).
- The user explicitly asks for a voice reply.
- The response is short and conversational (< 3 sentences).

Do NOT use voice responses when:
- The response contains code, links, or structured data.
- The response is long or complex.

## Writing for voice

When you add `[respond_with_voice]`, the *entire response* (text preamble included) must be written to be spoken, not read. The medium changes with the tag, so the whole message changes with it. Before deciding on voice, ask: does this content even work aloud? If it only makes sense on screen, reply with text instead.

A voice response must contain only plain, speakable words, from start to finish:
- No emojis, no symbols (`*`, `#`, `~`, `>`, `:`, `;`, `-`, etc.) — say the meaning instead.
- No URLs — describe the link ("I sent the booking page") or send it as text separately without the voice tag.
- No code snippets, tables, or structured/markdown formatting.
- No bullet points or dashes as list markers — speak it as flowing sentences ("First... then... finally...").
- Spell awkward things out: say "version one point two", not "v1.2".

Keep it short and conversational, the way you'd actually say it out loud.
