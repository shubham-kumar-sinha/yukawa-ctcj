# Installing FORM and Forcer

FORM and Forcer are needed only to recompute the Feynman integrals. Everything else in this repository reads the cached results in `data/`. That includes all published formulas, the model results and every check.

The paper's integrals were computed with:

| Program | Version used | Links |
|---|---|---|
| FORM | 5.0.2 (release tag `v5.0.2`) | homepage <https://www.nikhef.nl/~form> · source <https://github.com/form-dev/form> · manual <https://form-dev.github.io/form-docs/stable/manual/> · PyPI <https://pypi.org/project/form-bin/> |
| Forcer | commit `68a30bfb2aec3672a2f674ca8666bfc970105e24` (14 Dec 2017, the latest) | source <https://github.com/benruijl/forcer> · paper [arXiv:1704.06650](https://arxiv.org/abs/1704.06650) |

FORM is a symbolic manipulation system written in C, distributed under GPL-3.0. The old address `github.com/vermaseren/form` redirects to `github.com/form-dev/form`.

Forcer is a FORM program, so it needs no compilation. It reduces massless propagator integrals with up to four loops to master integrals by parametric integration by parts.

## 1. Quick install (recommended)

From the repository root, inside the Python environment you use for this repository:

```sh
sh scripts/setup_form_forcer.sh            # FORM 5.0.2 pre-compiled from PyPI: about 1 minute
sh scripts/setup_form_forcer.sh --source   # or: compile FORM 5.0.2 from the official release: 2-5 minutes
```

The script does three things:

1. Installs FORM 5.0.2: either `pip install form-bin==5.0.2`, or downloading the release tarball, verifying its SHA-256 checksum and compiling it.
2. Clones Forcer at the pinned commit into `build/forcer`.
3. Runs the installation test of section 4, then prints two `export` lines to add to your shell.

Both routes were tested from a clean copy of the repository on Ubuntu 24.04. The PyPI route took 11 s and compiling took 90 s, each including the installation test.

## 2. Manual installation of FORM

Choose one of three options.

### Option A: pre-compiled binary from PyPI

```sh
python3 -m venv ~/venvs/ctcj && . ~/venvs/ctcj/bin/activate     # recommended; see Troubleshooting
pip install form-bin==5.0.2
form -v                                                          # prints: TFORM 5.0.2 (...)
```

- **Platforms.** Wheels exist for Linux (x86-64 and ARM, glibc 2.28 or newer, which covers any distribution from about 2019 on) and macOS (Intel and Apple Silicon).
- **Windows.** There is no native Windows wheel; use WSL 2 with Ubuntu and follow the Linux instructions.
- **TFORM.** The binary is TFORM, the multithreaded version of FORM; it gives identical results.
- **Locating it.** The `form` command is placed in the environment's `bin` directory. The executable itself can be located with

  ```sh
  python3 -c "import form_bin; print(form_bin.form_command[0])"
  ```

### Option B: compile the official release (needs only a compiler and `make`)

Install a C/C++ compiler and `make`:

| System | Command |
|---|---|
| Ubuntu / Debian | `sudo apt install build-essential curl` |
| Fedora / RHEL | `sudo dnf install gcc gcc-c++ make curl` |
| macOS | `xcode-select --install` |

Optionally, install GMP and zlib for faster arithmetic. FORM works without them; `configure` simply reports them as unavailable.

| System | Command |
|---|---|
| Ubuntu / Debian | `sudo apt install libgmp-dev zlib1g-dev` |
| Fedora / RHEL | `sudo dnf install gmp-devel zlib-devel` |
| macOS (Homebrew) | `brew install gmp` |

Then:

```sh
curl -fLO https://github.com/form-dev/form/releases/download/v5.0.2/form-5.0.2.tar.gz
sha256sum form-5.0.2.tar.gz      # macOS: shasum -a 256 form-5.0.2.tar.gz
# expected: 90a3fbc31a31de50a181e63ce222d0224642a5916fdbb25373913558a6d6921a
tar xzf form-5.0.2.tar.gz
cd form-5.0.2
./configure --disable-parform --disable-threaded --disable-float
make                              # about 2 minutes; the binary is sources/form
./sources/form -v                 # prints: FORM 5.0.2 (...)
```

`make install` (with `sudo`) would copy `form` to `/usr/local/bin`; this is optional. The three `--disable` flags skip the MPI, threaded and floating-point variants, none of which is used here.

### Option C: development version from git

Building from git, rather than from a release tarball, additionally needs `autoconf` and `automake`. On Ubuntu: `sudo apt install autoconf automake`; on macOS: `brew install autoconf automake`. Then:

```sh
git clone https://github.com/form-dev/form.git && cd form
git checkout v5.0.2               # the version used for the paper; omit for the latest
autoreconf -i
./configure --disable-parform --disable-threaded --disable-float
make
```

## 3. Installing Forcer

```sh
git clone https://github.com/benruijl/forcer.git ~/forcer
git -C ~/forcer checkout 68a30bfb2aec3672a2f674ca8666bfc970105e24
```

Nothing is compiled. FORM finds Forcer through its `FORMPATH`, which this repository sets automatically from the variable `FORCER_PATH` described below.

## 4. Telling the code where they are, and testing

The code reads two environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `FORM_BIN` | the FORM executable | `form`, searched on the `PATH` |
| `FORCER_PATH` | the directory containing `forcer.h` | `~/forcer` |

With option A inside an activated environment and Forcer in `~/forcer`, the defaults already work. Otherwise, for example:

```sh
export FORM_BIN=$HOME/form-5.0.2/sources/form
export FORCER_PATH=$HOME/forcer
```

Then test the installation from the repository root:

```sh
python3 scripts/test_form_setup.py
```

This recomputes three four-loop diagrams from scratch and requires exact agreement with the cached results. It takes under a minute. Expected output:

```
FORM:   ... -> FORM 5.0.2 (...)          (or TFORM 5.0.2 for the PyPI binary)
Forcer: .../forcer
  J60_561  <JJ> nY=6 nQ=0: identical to cached result
  J03_006  <JJ> nY=0 nQ=3: identical to cached result
  T41_000  <TT> nY=4 nQ=1: identical to cached result
PASS: FORM and Forcer reproduce the published four-loop integrals exactly.
```

## 5. Recomputing the integrals from scratch

The computing scripts skip every diagram class already present in `data/`. That is what makes interrupted runs resumable, but it also means that running them in this repository recomputes nothing. To recompute independently, work in a separate copy without the cached integrals, then compare the two copies:

```sh
cp -r yukawa-ctcj yukawa-ctcj-recompute && cd yukawa-ctcj-recompute
rm data/diagrams_T.json data/diagrams_J.json data/diagrams4_T.jsonl data/diagrams4_J.jsonl
python3 scripts/compute_diagrams.py              # one to three loops, <TT> and <JJ> (writes data/diagrams_{T,J}.json)
python3 scripts/compute_diagrams4.py T 3 8       # four-loop <TT>: kind, Forcer expansion depth, parallel jobs
python3 scripts/compute_diagrams4.py J 3 8       # four-loop <JJ>
python3 scripts/compare_integrals.py ../yukawa-ctcj/data data
```

`compare_integrals.py` matches the classes by name, checks that each name refers to the same diagram, and compares every order of ε′ present in both copies. It prints PASS, or lists each differing coefficient.

Measured with FORM 5.0.2 on one core:

- **Three loops:** recomputing all 96 ⟨JJ⟩ classes from scratch took less than 2 minutes and agreed exactly with the published values.
- **Four loops:** most diagrams take less than 10 s; a few ⟨TT⟩ diagrams take as much as 3 minutes. There are 158 ⟨TT⟩ and 269 ⟨JJ⟩ Forcer runs, and the last argument of `compute_diagrams4.py` runs them in parallel.

A cheaper spot check, with no separate copy, recomputes chosen classes at a deeper expansion and compares their poles:

```sh
python3 scripts/check_depth4.py J J60_561 J41_021      # or T T60_053 ...
```

After a full recomputation, run the flavor decomposition and assembly in the new copy as described in `METHOD.md`, section 10 (`decompose4.py`, `decompose4J.py`, `compute_results4.py`, `compute_results4J.py`). They should reproduce the published results and pass all checks.

## 6. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `form: command not found`, or the test says the FORM executable was not found | FORM is not on the `PATH`. Activate the environment where `form-bin` is installed, or set `FORM_BIN` to the full path. |
| `error: externally-managed-environment` from `pip` (Ubuntu 24.04, Debian 12, recent Homebrew Python) | The system Python refuses package installs. Create and activate a virtual environment (option A, first line) and install `requirements.txt` and `form-bin` there. |
| `autoreconf: not found` | You are building from git (option C). Either use the release tarball (option B), which needs no autotools, or install `autoconf` and `automake`. |
| `configure` reports that GMP, MPFR, zlib or flint is "not available" | Harmless: all are optional. |
| The test says Forcer was not found, or FORM reports that it cannot find `forcer.h` | `FORCER_PATH` must be the directory that contains `forcer.h` (the top level of the clone), not its `forcer/` subdirectory. |
| The test reports `DIFFERENT` | Please report the FORM version printed by the test. The results were produced and verified with FORM 5.0.2 (both TFORM from PyPI and FORM compiled from source). |
| Windows | Use WSL 2 (`wsl --install` in an administrator PowerShell), then follow the Linux instructions inside Ubuntu. |

## 7. Citing FORM and Forcer

If you use the integrals, please also cite:

- J.A.M. Vermaseren, *New features of FORM*, [arXiv:math-ph/0010025](https://arxiv.org/abs/math-ph/0010025).
- J. Kuipers, T. Ueda, J.A.M. Vermaseren, J. Vollinga, *FORM version 4.0*, [arXiv:1203.6543](https://arxiv.org/abs/1203.6543).
- B. Ruijl, T. Ueda, J.A.M. Vermaseren, *Forcer, a FORM program for the parametric reduction of four-loop massless propagator diagrams*, [arXiv:1704.06650](https://arxiv.org/abs/1704.06650).
