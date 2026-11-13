"""Post review findings to a GitLab merge request as inline discussion threads.

Reads ``findings.json``, ``mr.json`` and ``diff.patch`` from a review directory produced by
``fetch_mr.sh``. Dry run by default: prints what would be sent. Pass ``--post`` to send.

Usage:
    python3 post_findings.py .mr-reviews/mr-42 [--post] [--summary] [--only B1,S2]

Inline comments go through ``glab api`` so authentication is whatever ``glab auth`` holds.
GitLab needs both ``old_line`` and ``new_line`` to anchor on an unchanged (context) line and
only ``new_line`` for an added line, so the diff is parsed to supply the right position.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def parse_new_side_lines(diff_text: str) -> dict[str, dict[int, int | None]]:
    """Map each file's new-side line numbers that appear in the diff to their old-side line.

    Args:
        diff_text: Unified diff as produced by ``git diff``.

    Returns:
        ``{new_path: {new_line: old_line_or_None}}``; ``None`` marks an added line.
    """
    anchors: dict[str, dict[int, int | None]] = {}
    current: dict[int, int | None] | None = None
    old_line = new_line = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            path = raw[4:]
            current = None if path == "/dev/null" else anchors.setdefault(path.removeprefix("b/"), {})
            continue
        if raw.startswith(("--- ", "diff --git")):
            continue
        header = HUNK_HEADER.match(raw)
        if header:
            old_line, new_line = int(header.group(1)), int(header.group(2))
            continue
        if current is None:
            continue
        if raw.startswith("+"):
            current[new_line] = None
            new_line += 1
        elif raw.startswith("-"):
            old_line += 1
        elif raw.startswith(" "):
            current[new_line] = old_line
            old_line += 1
            new_line += 1
    return anchors


def old_path_for(diff_text: str, new_path: str) -> str:
    """Return the pre-rename path of ``new_path`` (itself when the file was not renamed)."""
    match = re.search(rf"^rename from (.+)\nrename to {re.escape(new_path)}$", diff_text, re.MULTILINE)
    return match.group(1) if match else new_path


def glab_api(hostname: str, endpoint: str, fields: dict[str, str], send: bool) -> str:
    """Call ``glab api`` with POST form fields, or print the call when ``send`` is false."""
    command = ["glab", "api", "--hostname", hostname, "--method", "POST", endpoint]
    for key, value in fields.items():
        command += ["-f", f"{key}={value}"]
    if not send:
        return "(dry run)"
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return "posted"


def review_header(review_path: Path) -> str:
    """Return the review's verdict/intent/scope block: everything before the first ``## `` heading."""
    lines = review_path.read_text().splitlines()
    header: list[str] = []
    for line in lines:
        if line.startswith("## "):
            break
        header.append(line)
    return "\n".join(header).strip()


def main() -> int:
    """Validate findings against the diff and post (or preview) them."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("review_dir", type=Path)
    parser.add_argument("--post", action="store_true", help="Actually send to GitLab (default: dry run).")
    parser.add_argument("--summary", action="store_true", help="Also post the review header as a general note.")
    parser.add_argument("--only", help="Comma-separated finding IDs to include (default: all in findings.json).")
    args = parser.parse_args()

    mr_path = args.review_dir / "mr.json"
    if not mr_path.exists():
        print("ERROR: no mr.json — this was a branch-only review, so there is no MR to post to.", file=sys.stderr)
        return 2
    mr = json.loads(mr_path.read_text())
    findings = json.loads((args.review_dir / "findings.json").read_text())
    diff_text = (args.review_dir / "diff.patch").read_text()

    if args.only:
        wanted = {item.strip() for item in args.only.split(",")}
        findings = [finding for finding in findings if finding["id"] in wanted]

    refs = mr.get("diff_refs") or {}
    if not all(refs.get(key) for key in ("base_sha", "start_sha", "head_sha")):
        print("ERROR: mr.json has no diff_refs; re-run fetch_mr.sh.", file=sys.stderr)
        return 2
    hostname = urlparse(mr["web_url"]).hostname or "gitlab.com"
    endpoint_base = f"projects/{mr['project_id']}/merge_requests/{mr['iid']}"
    anchors = parse_new_side_lines(diff_text)

    mode = "POSTING" if args.post else "DRY RUN (pass --post to send)"
    print(f"{mode}: {len(findings)} finding(s) to {mr['web_url']}\n")

    failures = 0
    for finding in findings:
        path, line = finding["path"], int(finding["line"])
        file_anchors = anchors.get(path, {})
        if line not in file_anchors:
            print(f"[{finding['id']}] SKIPPED {path}:{line} — line is not in the diff, so GitLab cannot anchor it.")
            failures += 1
            continue
        fields = {
            "body": finding["body"],
            "position[position_type]": "text",
            "position[base_sha]": refs["base_sha"],
            "position[start_sha]": refs["start_sha"],
            "position[head_sha]": refs["head_sha"],
            "position[new_path]": path,
            "position[old_path]": old_path_for(diff_text, path),
            "position[new_line]": str(line),
        }
        old_line = file_anchors[line]
        if old_line is not None:
            fields["position[old_line]"] = str(old_line)
        try:
            status = glab_api(hostname, f"{endpoint_base}/discussions", fields, args.post)
        except RuntimeError as error:
            status = f"FAILED: {error}"
            failures += 1
        print(f"[{finding['id']}] {finding.get('severity', '?')} {path}:{line} — {status}")
        print("    " + finding["body"].replace("\n", "\n    ") + "\n")

    if args.summary:
        body = review_header(args.review_dir / "review.md")
        try:
            status = glab_api(hostname, f"{endpoint_base}/notes", {"body": body}, args.post)
        except RuntimeError as error:
            status = f"FAILED: {error}"
            failures += 1
        print(f"[summary] general note — {status}\n    " + body.replace("\n", "\n    "))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
