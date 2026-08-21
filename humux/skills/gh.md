# GitHub CLI (gh)

You have access to the `gh` CLI, installed and authenticated. Run it with the
`bash` tool for GitHub operations — issues, PRs, releases, repo/commit lookups.

## Read operations (pre-approved, run without asking)

```bash
gh issue list --repo owner/name
gh pr view 123 --repo owner/name
gh pr list --repo owner/name --state open
gh api repos/owner/name/commits
gh search issues "is:open label:bug" --repo owner/name
```

## Write operations (ask for confirmation first)

Creating issues, PRs, or releases (`gh issue create`, `gh pr create`,
`gh release create`, ...) asks the owner before running.

## Repo scoping

Always pass `--repo owner/name` unless the working directory is a checkout of
the target repository.

## Structured output

Use `--json <fields>` (e.g. `gh issue list --json number,title,labels`) or
`gh api` and parse the JSON when you need specific fields rather than reading
table output.

## Quoting Markdown bodies

Single-quote issue/PR bodies that contain Markdown (`--body '…```code```…'`):
single quotes make backticks and `$(…)` literal so the command guard keeps
them and the shell doesn't run them; double quotes leave them live and get
rejected by the command guard.

## Not allowed

`gh auth login/logout/setup-git/refresh` touch shared container auth state and
are always blocked — the agent's `gh` identity is wired in via env, not by
logging in/out.
