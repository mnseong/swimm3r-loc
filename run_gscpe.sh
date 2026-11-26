#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLD="${HOLD:-4}" bash "$D/run.sh" "${1:-v9c}" "${2:-out}" --gscpe "${@:3}"
