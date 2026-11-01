# usage: source ~/go2nav/env.sh   (do NOT add to .bashrc)
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate go2nav
export GO2NAV_HOME="$HOME/go2nav"
export CYCLONEDDS_HOME="$HOME/cyclonedds_ws/install/cyclonedds"          # 前任のビルド済み CycloneDDS 0.10.2 を読み取り専用で利用
export CYCLONEDDS_URI="file://$GO2NAV_HOME/config/cyclonedds.xml"
[ -f "$GO2NAV_HOME/config/secrets.env" ] && set -a && . "$GO2NAV_HOME/config/secrets.env" && set +a
# Jetson/aarch64: OpenCV fails with "libgomp.so.1: cannot allocate memory in static TLS block" unless libgomp is preloaded
[ "$(uname -m)" = "aarch64" ] && [ -f "$CONDA_PREFIX/lib/libgomp.so.1" ] && export LD_PRELOAD="$CONDA_PREFIX/lib/libgomp.so.1"
