#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOLD="${HOLD:-12}" BLOCK="${BLOCK:-3}" REF_MATCHER="${REF_MATCHER:-loftr}" TRT="${TRT:-1}" bash "$D/run.sh" "${1:-v17}" "${2:-out}" --gscpe "${@:3}"
