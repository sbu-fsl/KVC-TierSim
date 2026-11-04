# Load secrets.env (git-ignored) into the current shell. Source it, don't execute it:
#     source scripts/load_secrets.sh [path/to/secrets.env]
# Prints only the names of the variables it set, never their values.

_kv_file="${1:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)/secrets.env}"
if [ ! -f "$_kv_file" ]; then
    echo "load_secrets: $_kv_file not found (cp secrets.env.example secrets.env)" >&2
    return 1 2>/dev/null || exit 1
fi
set -a
. "$_kv_file"
set +a
# the fetcher's GitHub API calls read GITHUB_TOKEN; accept GH_TOKEN too
[ -n "${GH_TOKEN:-}" ] && [ -z "${GITHUB_TOKEN:-}" ] && export GITHUB_TOKEN="$GH_TOKEN"
echo "load_secrets: set $(grep -E '^[[:space:]]*(export[[:space:]]+)?[A-Za-z_][A-Za-z0-9_]*=.' "$_kv_file" \
    | sed -E 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/' | tr '\n' ' ')"
unset _kv_file
