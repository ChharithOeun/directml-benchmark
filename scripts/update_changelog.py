#!/usr/bin/env python3
"""update_changelog.py — auto-update CHANGELOG.md from git history."""

import subprocess, re, datetime, sys, argparse
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

CATEGORY_MAP = {
    "feat": "### Added", "fix": "### Fixed",
    "docs": "### Changed", "chore": "### Changed",
    "refactor": "### Changed", "perf": "### Added",
    "test": "### Added", "security": "### Security",
}

def git(*args):
    return subprocess.run(["git"] + list(args), capture_output=True, text=True,
                          cwd=REPO_ROOT).stdout.strip()

def get_new_commits():
    last = git("log", "--follow", "--pretty=format:%H", "--", "CHANGELOG.md").splitlines()
    log_range = f"{last[0]}..HEAD" if last else "HEAD~30..HEAD"
    raw = git("log", log_range, "--pretty=format:%h|%ad|%s|%an", "--date=short")
    if not raw:
        return []
    return [dict(zip(("sha","date","subject","author"), l.split("|",3)))
            for l in raw.splitlines() if "|" in l]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit",  action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commits = get_new_commits()
    if not commits:
        print("✅ Changelog up to date.")
        return

    print(f"📋 {len(commits)} new commit(s):")
    for c in commits:
        print(f"   {c['sha']} {c['subject']}")

    buckets = {}
    for c in commits:
        m = re.match(r'^(\w+)(\([^)]+\))?!?:\s*', c["subject"])
        prefix = m.group(1) if m else "chore"
        header = CATEGORY_MAP.get(prefix, "### Changed")
        scope_m = re.match(r'^\w+\(([^)]+)\)', c["subject"])
        scope = f"**{scope_m.group(1)}** — " if scope_m else ""
        body = re.sub(r'^\w+(\([^)]+\))?!?:\s*', '', c["subject"])
        buckets.setdefault(header, []).append(f"- {scope}{body} (`{c['sha']}`)")

    today = datetime.date.today().isoformat()
    block_lines = [f"## [Unreleased] — updated {today} ({len(commits)} commit(s))", ""]
    for h in ["### Added", "### Fixed", "### Changed", "### Security"]:
        if h in buckets:
            block_lines.append(h)
            block_lines.extend(buckets[h])
            block_lines.append("")
    block = "\n".join(block_lines) + "\n"

    if args.dry_run:
        print("\n--- Would inject ---")
        print(block)
        return

    content = CHANGELOG.read_text(encoding="utf-8")
    content = re.sub(r'## \[Unreleased\][^\n]*\n(?:(?!## \[).*\n)*', '', content)
    match = re.search(r'^## \[', content, re.MULTILINE)
    if match:
        content = content[:match.start()] + block + "\n" + content[match.start():]
    else:
        content = content.rstrip() + "\n\n" + block

    CHANGELOG.write_text(content, encoding="utf-8")
    print("✅ CHANGELOG.md updated.")

    if args.commit:
        subprocess.run(["git", "add", str(CHANGELOG)], cwd=REPO_ROOT)
        subprocess.run(["git", "commit", "-m", "chore(changelog): auto-update [skip ci]"],
                       cwd=REPO_ROOT)
        print("✅ Committed.")

if __name__ == "__main__":
    main()
