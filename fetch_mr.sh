#!/usr/bin/env bash
# Collects everything needed to review a merge request or branch into .mr-reviews/<slug>/
# and checks the MR head out into a detached worktree outside the repo.
#
# Usage:
#   fetch_mr.sh <MR number | !number | MR URL | branch> [--base <branch>]
#
# MR numbers/URLs need glab (authenticated). Branches only need git.
# The last line of output is the worktree path.
set -euo pipefail

usage() {
  echo "Usage: $0 <MR number | !number | MR URL | branch> [--base <branch>]" >&2
  exit 2
}

[ $# -ge 1 ] || usage
target="$1"; shift
base_override=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) [ $# -ge 2 ] || usage; base_override="$2"; shift 2 ;;
    *) usage ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"
repo_name="$(basename "$repo_root")"

json_field() {
  # Reads a dotted path (e.g. diff_refs.base_sha) from the JSON file given as $1.
  python3 -I -c '
import json, sys
value = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    value = value.get(key) if isinstance(value, dict) else None
print("" if value is None else value)
' "$1" "$2"
}

mode="branch"
mr_iid=""
repo_flag=()
if [[ "$target" =~ ^!?([0-9]+)$ ]]; then
  mode="mr"; mr_iid="${BASH_REMATCH[1]}"
elif [[ "$target" =~ ^(https?://[^/]+/.+)/-/merge_requests/([0-9]+) ]]; then
  mode="mr"; mr_iid="${BASH_REMATCH[2]}"; repo_flag=(-R "${BASH_REMATCH[1]}")
fi

if [ "$mode" = "mr" ]; then
  slug="mr-${mr_iid}"
else
  slug="branch-$(printf '%s' "$target" | tr -c 'A-Za-z0-9._-' '-')"
fi
out_dir="$repo_root/.mr-reviews/$slug"
mkdir -p "$out_dir"

if [ "$mode" = "mr" ]; then
  if ! command -v glab >/dev/null 2>&1; then
    echo "ERROR: '$target' looks like a merge request, but glab is not installed." >&2
    echo "Install and log in (brew install glab && glab auth login --hostname <gitlab host>)," >&2
    echo "or fetch the source branch yourself and pass the branch name instead." >&2
    exit 3
  fi
  glab mr view "$mr_iid" "${repo_flag[@]+"${repo_flag[@]}"}" -F json > "$out_dir/mr.json"

  title="$(json_field "$out_dir/mr.json" title)"
  author="$(json_field "$out_dir/mr.json" author.username)"
  source_branch="$(json_field "$out_dir/mr.json" source_branch)"
  target_branch="$(json_field "$out_dir/mr.json" target_branch)"
  web_url="$(json_field "$out_dir/mr.json" web_url)"
  description="$(json_field "$out_dir/mr.json" description)"
  [ -n "$base_override" ] && target_branch="$base_override"

  # Fetch from origin when it exists; otherwise straight from the project URL.
  fetch_from="origin"
  git remote get-url origin >/dev/null 2>&1 || fetch_from="${web_url%%/-/merge_requests/*}.git"
  # refs/merge-requests/<iid>/head exists on GitLab even for MRs from forks.
  git fetch --quiet "$fetch_from" \
    "+refs/merge-requests/${mr_iid}/head:refs/mr-review/${mr_iid}/head" \
    "+refs/heads/${target_branch}:refs/mr-review/${mr_iid}/target"
  head_ref="refs/mr-review/${mr_iid}/head"
  base_ref="refs/mr-review/${mr_iid}/target"
  label="!${mr_iid}"
else
  title="(branch review — no MR metadata)"
  author="$(git log -1 --format='%an' "$target" 2>/dev/null || echo unknown)"
  source_branch="$target"
  web_url=""
  description=""
  if git rev-parse --verify --quiet "$target^{commit}" >/dev/null; then
    head_ref="$target"
  elif git rev-parse --verify --quiet "origin/$target^{commit}" >/dev/null; then
    head_ref="origin/$target"
  else
    echo "ERROR: branch '$target' not found locally or as origin/$target. Fetch it first." >&2
    exit 4
  fi
  target_branch="${base_override:-main}"
  if git rev-parse --verify --quiet "$target_branch^{commit}" >/dev/null; then
    base_ref="$target_branch"
  elif git rev-parse --verify --quiet "origin/$target_branch^{commit}" >/dev/null; then
    base_ref="origin/$target_branch"
  else
    echo "ERROR: base branch '$target_branch' not found. Pass --base <branch>." >&2
    exit 4
  fi
  label="$target"
fi

head_sha="$(git rev-parse "$head_ref^{commit}")"
merge_base="$(git merge-base "$base_ref" "$head_sha")"

# Three-dot semantics: only what the MR adds, matching GitLab's "Changes" tab.
git diff --find-renames "$merge_base" "$head_sha" > "$out_dir/diff.patch"
git diff --find-renames --stat=200 "$merge_base" "$head_sha" > "$out_dir/diffstat.txt"
git diff --find-renames --name-status "$merge_base" "$head_sha" > "$out_dir/files.txt"
git log --format='%h %s (%an)' "$merge_base..$head_sha" > "$out_dir/commits.txt"

# Physical path (pwd -P): macOS TMPDIR sits behind the /var -> /private/var symlink, and
# `git worktree list` reports the resolved path, so comparing unresolved paths never matches.
mkdir -p "${TMPDIR:-/tmp}/mr-review-worktrees"
worktree="$(cd "${TMPDIR:-/tmp}/mr-review-worktrees" && pwd -P)/${repo_name}-${slug}"
git worktree prune
if git worktree list --porcelain | grep -qx "worktree $worktree"; then
  git -C "$worktree" checkout --quiet --detach "$head_sha"
else
  rm -rf "$worktree"
  mkdir -p "$(dirname "$worktree")"
  git worktree add --quiet --detach "$worktree" "$head_sha"
fi

{
  echo "# $label — $title"
  echo
  echo "- **Author:** $author"
  echo "- **Source:** \`$source_branch\` → **Target:** \`$target_branch\`"
  [ -n "$web_url" ] && echo "- **URL:** $web_url"
  echo "- **Head SHA:** \`$head_sha\`"
  echo "- **Merge base:** \`$merge_base\`"
  echo "- **Worktree:** \`$worktree\`"
  echo "- **Commits:** $(wc -l < "$out_dir/commits.txt" | tr -d ' ')"
  echo "- **Files changed:** $(wc -l < "$out_dir/files.txt" | tr -d ' ')"
  echo
  echo "## Description"
  echo
  if [ -n "$description" ]; then echo "$description"; else echo "_(none)_"; fi
  echo
  echo "## Diffstat"
  echo
  echo '```'
  cat "$out_dir/diffstat.txt"
  echo '```'
} > "$out_dir/meta.md"

echo "Review data: $out_dir"
echo "Worktree: $worktree"
