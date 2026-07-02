# Weather — curl wttr.in

You can fetch weather information using `curl` and the `wttr.in` service.

## Current weather

```bash
# Current weather for a city (concise one-line output)
curl -s "wttr.in/London?format=3"

# Current weather with more detail (compact)
curl -s "wttr.in/London?format=%C+%t+%w+%h+%p\n"
```

## Forecast

```bash
# Full 3-day forecast (terminal-formatted)
curl -s "wttr.in/London"

# JSON forecast (machine-readable, use with jq)
curl -s "wttr.in/London?format=j1"

# One-day forecast only
curl -s "wttr.in/London?1"
```

## JSON output with jq

```bash
# Current temperature and condition
curl -s "wttr.in/London?format=j1" | jq '.current_condition[0] | {temp_C, weatherDesc: .weatherDesc[0].value, humidity, windspeedKmph}'

# Today's min/max temperature
curl -s "wttr.in/London?format=j1" | jq '.weather[0] | {date, mintempC, maxtempC}'

# 3-day summary
curl -s "wttr.in/London?format=j1" | jq '.weather[] | {date, mintempC, maxtempC, desc: .hourly[4].weatherDesc[0].value}'
```

## Format placeholders

Use `?format=` for custom one-line output:

| Placeholder | Description |
|---|---|
| `%C` | Weather condition text |
| `%t` | Temperature |
| `%f` | Feels-like temperature |
| `%w` | Wind |
| `%h` | Humidity |
| `%p` | Precipitation (mm) |
| `%P` | Pressure |
| `%l` | Location |

Example: `curl -s "wttr.in/Zurich?format=%l:+%C+%t+(feels+like+%f)+%w"`

## Important notes

- Always use `-s` (silent) to suppress curl's progress bar.
- URL-encode spaces in city names: `curl -s "wttr.in/New+York?format=3"`
- For GPS coordinates: `curl -s "wttr.in/47.37,8.55?format=j1"`
- Use `?format=j1` when you need to extract specific data points — pipe through `jq`.
- Use `?format=3` for a quick one-line answer when the user just asks "what's the weather".
- Default to the owner's configured timezone/location when no city is specified.
