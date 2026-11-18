#!/bin/sh
# Install FORM and Forcer for this repository; see INSTALL_FORM.md for details and manual instructions.
#
#   sh scripts/setup_form_forcer.sh            # FORM 5.0.2 pre-compiled from PyPI (Linux, macOS; ~1 min)
#   sh scripts/setup_form_forcer.sh --source   # FORM 5.0.2 compiled from the official release tarball (~2-5 min;
#                                              #   needs only a C/C++ compiler and make)
#
# Both routes put Forcer (pinned commit) into ./build/forcer, run scripts/test_form_setup.py, and print the two
# environment variables to export.  Run from the repository root.
set -e
FORM_VERSION=5.0.2
FORM_SHA256=90a3fbc31a31de50a181e63ce222d0224642a5916fdbb25373913558a6d6921a
FORM_URL=https://github.com/form-dev/form/releases/download/v$FORM_VERSION/form-$FORM_VERSION.tar.gz
FORCER_URL=https://github.com/benruijl/forcer.git
FORCER_COMMIT=68a30bfb2aec3672a2f674ca8666bfc970105e24
PY=${PYTHON:-python3}

[ -f ctcj/forcer.py ] || { echo "run this from the repository root"; exit 1; }
mkdir -p build
ROOT=$(pwd)

# ------------------------------------------------------------------ FORM
if [ "$1" = "--source" ]; then
    cd build
    if [ ! -x form-$FORM_VERSION/sources/form ]; then
        echo "== downloading $FORM_URL"
        curl -fL -o form-$FORM_VERSION.tar.gz "$FORM_URL"
        if command -v sha256sum > /dev/null; then SUM=$(sha256sum form-$FORM_VERSION.tar.gz | cut -d' ' -f1)
        else SUM=$(shasum -a 256 form-$FORM_VERSION.tar.gz | cut -d' ' -f1); fi
        [ "$SUM" = "$FORM_SHA256" ] || { echo "checksum mismatch for form-$FORM_VERSION.tar.gz"; exit 1; }
        tar xzf form-$FORM_VERSION.tar.gz
        cd form-$FORM_VERSION
        echo "== configuring and compiling FORM (a few minutes)"
        ./configure --disable-parform --disable-threaded --disable-float > configure.log 2>&1 \
            || { echo "configure failed, see build/form-$FORM_VERSION/configure.log"; exit 1; }
        make > make.log 2>&1 || { echo "make failed, see build/form-$FORM_VERSION/make.log"; exit 1; }
        cd ..
    fi
    FORM_BIN=$ROOT/build/form-$FORM_VERSION/sources/form
    cd "$ROOT"
else
    echo "== installing FORM $FORM_VERSION from PyPI (package form-bin)"
    $PY -m pip install "form-bin==$FORM_VERSION"
    FORM_BIN=$($PY -c "import form_bin; print(form_bin.form_command[0])")
fi
"$FORM_BIN" -v | head -1

# ------------------------------------------------------------------ Forcer
if [ ! -f build/forcer/forcer.h ]; then
    echo "== cloning Forcer"
    git clone -q "$FORCER_URL" build/forcer
    (cd build/forcer && git checkout -q $FORCER_COMMIT)
fi
FORCER_PATH=$ROOT/build/forcer

# ------------------------------------------------------------------ test
export FORM_BIN FORCER_PATH
$PY scripts/test_form_setup.py
echo
echo "Add these two lines to your shell (or ~/.bashrc / ~/.zshrc) before running the Forcer scripts:"
echo "export FORM_BIN=$FORM_BIN"
echo "export FORCER_PATH=$FORCER_PATH"
