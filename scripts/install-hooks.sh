#!/usr/bin/env bash
# Install the privacy check as a git pre-commit and pre-push hook.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
for h in pre-commit pre-push; do
  printf '#!/usr/bin/env bash\nexec "%s" --all\n' '$(git rev-parse --show-toplevel)/scripts/privacy-check.sh' > "$root/.git/hooks/$h"
  chmod +x "$root/.git/hooks/$h"
done
echo "  privacy hooks installed (pre-commit, pre-push)"
