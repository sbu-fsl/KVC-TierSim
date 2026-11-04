# Activate the project environment in the current shell. Source it, don't execute it:
#     source scripts/activate.sh
# - activates the venv (VENV from local.env, else .venv)
# - exports KVWL_DATA_DIR from local.env (DATA_DIR), so `python -m kvwl...` uses the data store
# - loads secrets.env (prints variable names only)

_kv_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_kv_venv="$_kv_repo/.venv"
if [ -f "$_kv_repo/local.env" ]; then
    _kv_data="$(sed -nE 's/^[[:space:]]*DATA_DIR[[:space:]]*:?=[[:space:]]*//p' "$_kv_repo/local.env" | tail -1)"
    _kv_v="$(sed -nE 's/^[[:space:]]*VENV[[:space:]]*:?=[[:space:]]*//p' "$_kv_repo/local.env" | tail -1)"
    [ -n "$_kv_v" ] && _kv_venv="$_kv_v"
    [ -n "$_kv_data" ] && export KVWL_DATA_DIR="$_kv_data"
fi
if [ -f "$_kv_venv/bin/activate" ]; then
    . "$_kv_venv/bin/activate"
else
    echo "activate: no venv at $_kv_venv (run scripts/setup_host.sh or make venv)" >&2
fi
[ -f "$_kv_repo/secrets.env" ] && . "$_kv_repo/scripts/load_secrets.sh" "$_kv_repo/secrets.env"
echo "activate: venv=$_kv_venv KVWL_DATA_DIR=${KVWL_DATA_DIR:-<unset: datasets/data>}"
unset _kv_repo _kv_venv _kv_data _kv_v
