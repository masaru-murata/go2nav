#!/bin/bash
# Deploy go2nav/ to the Jetson's ~/go2nav (home-only, no sudo) and install the package into the conda env.
# usage: scripts/deploy_go2nav.sh [host]   (default ubuntu.local over Wi-Fi; wired: 192.168.123.18. The static 10.9.255.250 is not reachable through the guest AP, use the mDNS name or the DHCP address)
set -e
HOST=${1:-ubuntu.local}
SRC="$(cd "$(dirname "$0")/.." && pwd)/go2nav/"
rsync -av --exclude third_party --exclude data --exclude '__pycache__' --exclude '*.egg-info' --exclude '.pytest_cache' \
      --exclude 'config/secrets.env' "$SRC" unitree@$HOST:~/go2nav/
ssh unitree@$HOST 'source ~/go2nav/env.sh && python -c "import pytest" 2>/dev/null || pip install -q pytest; source ~/go2nav/env.sh && pip install -q -e ~/go2nav && go2nav check; echo; echo "run tests: source ~/go2nav/env.sh && cd ~/go2nav && python -m pytest tests -q"'
