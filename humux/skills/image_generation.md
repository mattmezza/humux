# Image generation — generate_image

Generate an image from a text prompt and deliver it to the user as a native
photo. Use the `generate_image` tool (not `run_command`) whenever the user asks
for a picture, illustration, diagram, concept art, logo, sticker, or any visual.

The tool sends the image to the user automatically. **Do not** paste the file
path or base64 into your reply — just say briefly what you made (e.g. "Here's the
sunset over mountains you asked for"). The image arrives as a separate photo.

## How to call it

```
generate_image(prompt="a watercolor fox curled asleep under autumn leaves, soft warm light")
generate_image(prompt="minimalist flat-vector logo of a paper plane, single teal accent", size="1024x1024")
```

- `prompt` (required) — describe the image in detail. The more specific, the better.
- `size` (optional) — `WIDTHxHEIGHT` like `1024x1024`, `1536x1024` (landscape),
  `1024x1536` (portrait). Honored by OpenAI; OpenRouter/fal use the model default.

## Writing good prompts

- Name the **subject**, then **style**, then **composition/lighting/mood**.
  e.g. "a red panda astronaut *(subject)*, retro sci-fi poster *(style)*,
  centered, dramatic rim light, deep blue background *(composition)*".
- Be concrete about medium: photo, watercolor, oil painting, 3D render, flat
  vector, pixel art, line drawing, blueprint.
- For **text in an image** (signs, labels, logos), keep the words short and put
  them in quotes. OpenAI's GPT Image models render text far better than diffusion
  models (Flux); if text accuracy matters and the provider is OpenAI, expect good
  results — on Flux providers, minimize embedded text.
- For diagrams/flowcharts, describe boxes, arrows, and labels explicitly.
- Negatives help: "no text, no watermark, no border".

## Providers & trade-offs

The provider and model are set by the admin in config — you don't choose them.
For reference:

| Provider | Default model | Strength |
|---|---|---|
| OpenRouter | Gemini 2.5 Flash Image (Nano Banana) | Cheap, zero new auth (reuses LLM key) |
| fal.ai | Flux Schnell | Cheapest (~$0.003), sub-second |
| OpenAI | GPT Image Mini | Best text-in-image, strong prompt understanding |

## Notes & limits

- A daily/monthly budget cap may be set. If it's reached, the tool returns a
  budget error — tell the user the cap was hit and that it resets (daily caps at
  00:00 UTC); don't retry in a loop.
- If the tool reports image generation is disabled, tell the user they can enable
  it in the admin settings (Settings → Tools → Image generation).
- Generation can take from under a second (Flux Schnell) to ~10-15s (GPT Image);
  that wait is normal.
