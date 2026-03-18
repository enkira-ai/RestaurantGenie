#!/usr/bin/env bash
# Usage: ./predict.sh "123 Main St, Austin TX" italian 2
#
# Arguments:
#   $1  Full US address (quoted)
#   $2  Cuisine type: american, burgers, chinese, french, greek, indian, italian,
#                     japanese, korean, mediterranean, mexican, other, pizza,
#                     sandwiches, seafood, steakhouses, thai, vietnamese
#   $3  Price level: 1=$ 2=$$ 3=$$$ 4=$$$$

set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: ./predict.sh \"<address>\" <cuisine> <price>"
    echo ""
    echo "  address   Full US address, e.g. \"123 Main St, Austin TX\""
    echo "  cuisine   e.g. italian, mexican, american, pizza, chinese, japanese ..."
    echo "  price     1=\$  2=\$\$  3=\$\$\$  4=\$\$\$\$"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run --project "$SCRIPT_DIR" python -W ignore -m src.predict \
    --address "$1" \
    --cuisine "$2" \
    --price "$3"
