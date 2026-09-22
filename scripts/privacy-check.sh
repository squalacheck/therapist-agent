#!/usr/bin/env bash
# ---------------------------------------------------------------------
# Refuse to publish anything personal.
#
# Runs as a pre-commit hook (scripts/install-hooks.sh) and on demand:
#     scripts/privacy-check.sh            check the files git would commit
#     scripts/privacy-check.sh --all      check every file in the tree
#
# Two layers:
#   1. Generic rules that apply to anyone: no .env, no data files, no
#      database dumps, no absolute home paths, no real email addresses.
#   2. A personal denylist — names, places, anything that identifies you —
#      kept OUTSIDE the repo, because the list itself is personal:
#          ~/.config/therapist-agent-private/denylist.txt
#      (override with PRIVACY_DENYLIST=/path). One term per line, matched
#      case-insensitively on word boundaries. Lines starting # are ignored.
# ---------------------------------------------------------------------
set -uo pipefail
# Always the repo this script lives in, never the caller's directory. An
# earlier version resolved to the repo's parent and scanned all of $HOME.
cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
[ -f install.sh ] && [ -d services/rag ] || { echo "  privacy check: not in the therapist-agent repo"; exit 1; }

DENY="${PRIVACY_DENYLIST:-$HOME/.config/therapist-agent-private/denylist.txt}"
red=$'\033[31m'; grn=$'\033[32m'; off=$'\033[0m'
fail=0
flag() { echo "${red}  BLOCK${off} $*"; fail=1; }

if [ "${1:-}" = "--all" ] || ! git rev-parse --git-dir >/dev/null 2>&1; then
  mapfile -t files < <(find . -type f -not -path './.git/*' | sed 's|^\./||')
else
  mapfile -t files < <(git diff --cached --name-only --diff-filter=ACMR)
  [ ${#files[@]} -eq 0 ] && mapfile -t files < <(git ls-files)
fi

# 1. Files that must never be in the repo at all.
for f in "${files[@]}"; do
  case "$f" in
    .env|*/.env|.env.local|*.jsonl|*.dump|*.sql.gz|*.sqlite|*.db|*.pdf|data/processed/*|data/raw/direct/*|backups/*)
      flag "$f — this kind of file must never be committed" ;;
  esac
done

# Text files only from here on.
text=()
for f in "${files[@]}"; do
  [ -f "$f" ] && grep -Iq . "$f" 2>/dev/null && text+=("$f")
done
[ ${#text[@]} -eq 0 ] && { echo "${grn}  privacy check: nothing to scan${off}"; exit $fail; }

# 2. Generic identifiers.
hits=$(grep -nH -E '/(home|Users)/[A-Za-z0-9._-]+' "${text[@]}" | grep -v -E '/(home|Users)/(user|you|me|runner|<[^>]*>)\b' || true)
[ -n "$hits" ] && { flag "absolute home-directory paths:"; echo "$hits" | sed 's/^/        /'; }

hits=$(grep -nH -o -E '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' "${text[@]}" \
  | grep -v -i -E '@(example\.(com|org)|users\.noreply\.github\.com|anthropic\.com)$|noreply@' \
  | grep -v -E ':(git@github\.com|[a-z]+@[0-9])' || true)
[ -n "$hits" ] && { flag "email addresses:"; echo "$hits" | sed 's/^/        /'; }

# 3. The personal denylist.
if [ -f "$DENY" ]; then
  n=0
  while IFS= read -r term || [ -n "$term" ]; do
    term="${term%%#*}"; term="$(echo "$term" | sed 's/[[:space:]]*$//; s/^[[:space:]]*//')"
    [ -z "$term" ] && continue
    n=$((n+1))
    hits=$(grep -nH -i -w -F -- "$term" "${text[@]}" || true)
    [ -n "$hits" ] && { flag "denylisted term \"$term\":"; echo "$hits" | cut -c1-200 | sed 's/^/        /'; }
  done < "$DENY"
  echo "  checked ${#text[@]} files against $n personal terms"
else
  echo "  note: no personal denylist at $DENY — only generic rules applied"
fi

if [ $fail -eq 0 ]; then echo "${grn}  privacy check passed${off}"; else
  echo; echo "${red}  privacy check FAILED — nothing personal may be committed.${off}"; fi
exit $fail
