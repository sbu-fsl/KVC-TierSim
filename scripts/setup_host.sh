#!/usr/bin/env bash
# Set up a host (e.g. ibm1.fsl) after cloning the repo: Python env, permanent data store,
# credentials, raw datasets, request streams. Safe to re-run: finished steps are skipped
# (downloads keep existing files, the venv is only rebuilt when requirements change).
#
#   git clone <repo> kvc_tiersim_kvwl && cd kvc_tiersim_kvwl
#   scripts/setup_host.sh                                   # data store: /mnt/gpfs/kvc_tiersim_kvwl
#   scripts/setup_host.sh --data-dir /some/other/place --python python3.11
#
# Options
#   --data-dir DIR    permanent data store for raw/, streams/, .cache/ (default: $KVWL_DATA_DIR,
#                     else the previous run's local.env, else /mnt/gpfs/kvc_tiersim_kvwl)
#   --python BIN      interpreter for the venv (default: python3; needs >= 3.10)
#   --venv DIR        virtualenv location (default: .venv in the repo)
#   --secrets FILE    copy credentials (HF_TOKEN=..., GH_TOKEN=...) from FILE into secrets.env
#   --hf-streams      also rebuild the `published` workloads from Hugging Face data (make profile-hf)
#   --skip-fetch      do not download raw sources
#   --skip-profile    do not build request streams
#   --dry-run         print the steps without running them
#   -h, --help
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# defaults: a previous run's local.env, then $KVWL_DATA_DIR, then ibm1.fsl's GPFS
_prev() { [ -f "$REPO/local.env" ] && sed -nE "s/^[[:space:]]*$1[[:space:]]*:?=[[:space:]]*//p" "$REPO/local.env" | tail -1; }
DATA_DIR="${KVWL_DATA_DIR:-$(_prev DATA_DIR)}"; DATA_DIR="${DATA_DIR:-/mnt/gpfs/kvc_tiersim_kvwl}"
PYTHON_BIN="$(_prev PYTHON)"; PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$(_prev VENV)"; VENV_DIR="${VENV_DIR:-$REPO/.venv}"
SECRETS_SRC=""
HF_STREAMS=0
SKIP_FETCH=0
SKIP_PROFILE=0
DRY=0
NEED_GB=3   # raw ~1.3 GB + download cache ~0.1 GB + streams ~0.2 GB, with headroom

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d; s/^# \{0,1\}//'; }
while [ $# -gt 0 ]; do
    case "$1" in
        --data-dir)     DATA_DIR="$2"; shift 2 ;;
        --python)       PYTHON_BIN="$2"; shift 2 ;;
        --venv)         VENV_DIR="$2"; shift 2 ;;
        --secrets)      SECRETS_SRC="$2"; shift 2 ;;
        --hf-streams)   HF_STREAMS=1; shift ;;
        --skip-fetch)   SKIP_FETCH=1; shift ;;
        --skip-profile) SKIP_PROFILE=1; shift ;;
        --dry-run)      DRY=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

cd "$REPO"
DATA_DIR="${DATA_DIR%/}"
MAKE_VARS=(VENV="$VENV_DIR" PYTHON="$PYTHON_BIN" DATA_DIR="$DATA_DIR")
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
run()  { echo "+ $*"; [ "$DRY" = 1 ] || "$@"; }
die()  { echo "setup_host: $*" >&2; exit 1; }

# ------------------------------------------------------------------ 1. prerequisites
step "Checking prerequisites"
for tool in git make curl; do
    command -v "$tool" >/dev/null || die "'$tool' not found"
done
command -v "$PYTHON_BIN" >/dev/null || die "'$PYTHON_BIN' not found (use --python)"
"$PYTHON_BIN" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
    || die "$PYTHON_BIN is $("$PYTHON_BIN" -V 2>&1); need Python >= 3.10 (use --python)"
"$PYTHON_BIN" -c 'import venv, ensurepip' 2>/dev/null \
    || die "$PYTHON_BIN lacks venv/ensurepip (install python3-venv or use another --python)"
echo "repo:       $REPO"
echo "python:     $("$PYTHON_BIN" -V 2>&1) ($(command -v "$PYTHON_BIN"))"
echo "venv:       $VENV_DIR"
echo "data store: $DATA_DIR"

# ------------------------------------------------------------------ 2. data store
step "Preparing the data store"
parent="$(dirname "$DATA_DIR")"
[ -d "$parent" ] || die "$parent does not exist (is the file system mounted?)"
run mkdir -p "$DATA_DIR/raw" "$DATA_DIR/streams" "$DATA_DIR/.cache"
if [ "$DRY" = 0 ]; then
    [ -w "$DATA_DIR" ] || die "$DATA_DIR is not writable"
    free_kb=$(df -Pk "$DATA_DIR" | awk 'NR==2 {print $4}')
    if [ -n "$free_kb" ] && [ "$free_kb" -lt $((NEED_GB * 1024 * 1024)) ]; then
        die "only $((free_kb / 1024 / 1024)) GB free in $DATA_DIR; need ~${NEED_GB} GB"
    fi
    echo "free space: $((free_kb / 1024 / 1024)) GB"
fi

# ------------------------------------------------------------------ 3. per-host config
step "Writing local.env (git-ignored; read by make and scripts/activate.sh)"
if [ "$DRY" = 1 ]; then
    echo "+ write local.env: DATA_DIR=$DATA_DIR VENV=$VENV_DIR PYTHON=$PYTHON_BIN"
else
    cat > local.env <<EOF
# Written by scripts/setup_host.sh on $(hostname) at $(date '+%Y-%m-%d %H:%M'). Plain KEY=VALUE (make syntax).
DATA_DIR=$DATA_DIR
VENV=$VENV_DIR
PYTHON=$PYTHON_BIN
EOF
    echo "local.env: DATA_DIR=$DATA_DIR"
fi

# ------------------------------------------------------------------ 4. credentials
step "Credentials (secrets.env, git-ignored)"
if [ -n "$SECRETS_SRC" ]; then
    [ -f "$SECRETS_SRC" ] || die "--secrets: $SECRETS_SRC not found"
    run cp "$SECRETS_SRC" secrets.env
    run chmod 600 secrets.env
elif [ ! -f secrets.env ]; then
    run cp secrets.env.example secrets.env
    run chmod 600 secrets.env
    if [ "$DRY" = 0 ] && [ -t 0 ]; then
        printf 'HF_TOKEN (input hidden; Enter to skip): '
        read -rs token; echo
        if [ -n "$token" ]; then
            tmp="$(mktemp)"; sed "s|^HF_TOKEN=.*|HF_TOKEN=$token|" secrets.env > "$tmp" && mv "$tmp" secrets.env && chmod 600 secrets.env
        fi
        unset token
    fi
fi
if [ "$DRY" = 0 ]; then
    if grep -qE '^[[:space:]]*(export[[:space:]]+)?HF_TOKEN=[^[:space:]]' secrets.env 2>/dev/null; then
        echo "HF_TOKEN: set"
    else
        echo "HF_TOKEN: missing -> the gated datasets (LMSYS-Chat-1M, GPQA) will be skipped; edit secrets.env and re-run"
    fi
fi

# ------------------------------------------------------------------ 5. python environment
step "Python environment (make venv)"
run make --no-print-directory "${MAKE_VARS[@]}" venv
run make --no-print-directory "${MAKE_VARS[@]}" smoke

# ------------------------------------------------------------------ 6. datasets
if [ "$SKIP_FETCH" = 0 ]; then
    step "Raw sources -> $DATA_DIR/raw (make fetch, make fetch-hf)"
    run make --no-print-directory "${MAKE_VARS[@]}" fetch
    # hf sources that fail (e.g. no token for a gated set) are reported, not fatal
    run make --no-print-directory "${MAKE_VARS[@]}" fetch-hf || echo "!! some Hugging Face sources failed; see above (re-run to retry)"
fi

if [ "$SKIP_PROFILE" = 0 ]; then
    step "Request streams -> $DATA_DIR/streams (make profile)"
    run make --no-print-directory "${MAKE_VARS[@]}" profile
    if [ "$HF_STREAMS" = 1 ]; then
        step "Rebuilding \`published\` workloads from Hugging Face data (make profile-hf)"
        run make --no-print-directory "${MAKE_VARS[@]}" profile-hf
    fi
fi

# ------------------------------------------------------------------ done
step "Done"
[ "$DRY" = 0 ] && run make --no-print-directory "${MAKE_VARS[@]}" data-dir
cat <<EOF

Next:
  source scripts/activate.sh     # venv + KVWL_DATA_DIR + secrets in this shell
  make help                      # all tasks; make picks up local.env automatically
EOF
