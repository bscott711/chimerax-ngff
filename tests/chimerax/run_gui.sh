#!/bin/sh
# Run ChimeraX on a private Xvfb with OpenGL on a GPU through VirtualGL, for GUI timing
# and checks that need a real GL context without touching anyone's desktop.
#
#   tests/chimerax/run_gui.sh [chimerax arguments...]
#
# Environment:
#   NGFF_GUI_HOME     HOME for ChimeraX (default: a fresh dir under $TMPDIR), so its user
#                     site and settings stay apart from your own
#   NGFF_GUI_GL       software (default, Mesa llvmpipe) or vgl (VirtualGL; see below)
#   NGFF_GUI_GPU      VirtualGL EGL device for vgl (default egl1: GPU 1, off the decon GPU)
#   NGFF_GUI_DISPLAY  X display number (default 97)
#   NGFF_GUI_SIZE     screen size (default 1700x1000x24)
#   ARGUS_CHIMERAX    ChimeraX binary (default: the shared Argus install)
# Needs the Xvfb and VirtualGL unpacked by naparym-live under ~/opt/naparym-live/current/sysroot.
set -eu
sysroot=${NGFF_SYSROOT:-$HOME/opt/naparym-live/current/sysroot}
chimerax=${ARGUS_CHIMERAX:-$HOME/software/chimerax/usr/bin/chimerax-daily}
gl=${NGFF_GUI_GL:-software}
work=$(mktemp -d "${TMPDIR:-/tmp}/ngff-gui.XXXXXX")
gui_home=${NGFF_GUI_HOME:-$work/home}
mkdir -p "$gui_home"
trap 'kill ${xvfb:-} 2>/dev/null; rm -rf "$work"' EXIT INT TERM
if [ "$gl" = dcv ]; then
    # The DCV desktop: real NVIDIA GLX. The display number changes when the session is
    # recreated, so read it from the DCV agent.
    auth=${XAUTHORITY:-/run/user/$(id -u)/dcv/bioimaging.xauth}
    num=${NGFF_GUI_DISPLAY:-$(ps -eo args | sed -n 's/.*dcvagent .*--session-id bioimaging .*--display :\([0-9]*\).*/\1/p' | head -1)}
else
    num=${NGFF_GUI_DISPLAY:-97}
    auth=$work/xauth
    touch "$auth"
    cookie=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
    xauth -f "$auth" add ":$num" . "$cookie"
    "$sysroot/usr/bin/Xvfb" ":$num" -screen 0 "${NGFF_GUI_SIZE:-1700x1000x24}" -auth "$auth" \
        -nolisten tcp -noreset >"$work/xvfb.log" 2>&1 &
    xvfb=$!
    sleep 1
fi

# NGFF_GUI_GL=dcv: your DCV desktop with its NVIDIA GPU (a window opens there; use a private
# HOME, as always). This is the one that measures the GPU.
# NGFF_GUI_GL=software (default): Mesa llvmpipe on the CPU. It exercises everything ChimeraX
# does per frame except the GPU itself (data reads, colour, blend, texture fill, histogram).
# NGFF_GUI_GL=vgl: VirtualGL on a GPU; ChimeraX's default framebuffer does not work on it yet.
if [ "$gl" = dcv ]; then
    gl_env="PYOPENGL_PLATFORM=glx"
elif [ "$gl" = vgl ]; then
    gl_env="LD_PRELOAD=$sysroot/usr/lib64/VirtualGL/libvglfaker-nodl.so VGL_DISPLAY=${NGFF_GUI_GPU:-egl1} VGL_EGLLIB=/usr/lib64/libEGL.so.1 VGL_GLLIB=/usr/lib64/libGL.so.1 LD_LIBRARY_PATH=$sysroot/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} PYOPENGL_PLATFORM=egl"
else
    gl_env="LIBGL_ALWAYS_SOFTWARE=1 PYOPENGL_PLATFORM=glx"
fi
# shellcheck disable=SC2086
env HOME=$gui_home DISPLAY=:$num XAUTHORITY=$auth QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu \
    $gl_env "$chimerax" "$@"
