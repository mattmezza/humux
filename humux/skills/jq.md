# jq — JSON Processor

You have access to `jq` for filtering, transforming, and extracting data from JSON output.
Use it when you need to pull specific fields from large JSON responses rather than
passing the entire blob back to the user.

## Piping JSON from other tools

```bash
# Extract just subjects from himalaya output
himalaya -a personal envelope list -s 10 -o json | jq '.[].subject'

# Get emails from a specific sender
himalaya -a work envelope list -s 20 -o json | jq '[.[] | select(.from.addr | test("example.com"))]'

# Extract calendar event summaries and times
python3 /app/tools/calendar_read.py --calendar google --today -o json | jq '.[] | {summary, start, end}'

# Count results
himalaya -a personal envelope list -o json | jq 'length'
```

## Standalone usage

```bash
# Parse a JSON string
echo '{"name": "Alice", "age": 30}' | jq '.name'

# Pretty-print JSON
echo '[1,2,3]' | jq '.'
```

## Useful filters

| Filter | Description |
|---|---|
| `.field` | Extract a field |
| `.[]` | Iterate array elements |
| `.[0]` | First element |
| `select(condition)` | Filter elements |
| `{a, b}` | Pick specific keys |
| `map(expr)` | Transform each element |
| `length` | Count elements |
| `test("regex")` | Regex match |
| `keys` | List object keys |
| `sort_by(.field)` | Sort array by field |

## Important notes

- Use `-r` for raw string output (no quotes): `jq -r '.name'`
- Use `-e` to set exit code based on output (useful for conditionals).
- Prefer jq over asking the LLM to parse JSON mentally — it's faster and more reliable.
