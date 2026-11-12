# Installing ATESTool

**Use Python 3.12 for the examples.** Conda installs it inside the ATESTool
environment. Choose the setup for your platform. The commands under
[Install and verify](#install-and-verify) are shared across all three platforms.

- [Windows](#windows)
- [Linux](#linux)
- [macOS](#macos)

## Windows

1. If needed, [install Miniconda for Windows](https://docs.conda.io/projects/conda/en/stable/user-guide/install/windows.html).
   Open **Anaconda Prompt** from the Start menu.
2. Run `git --version`. If Git is missing, [install Git for Windows](https://git-scm.com/install/windows),
   reopen Anaconda Prompt, and repeat the check.
3. [Create the environment, install ATESTool, and run the example](#install-and-verify).

Use `dir` to list the current folder.

## Linux

1. If needed, [install Miniconda for Linux](https://docs.conda.io/projects/conda/en/stable/user-guide/install/linux.html).
   Initialize Conda for your shell when prompted, then open a new terminal.
2. Run `git --version`. If Git is missing, use `sudo apt install git` on Debian or
   Ubuntu, or [follow the instructions for your distribution](https://git-scm.com/install/linux).
3. [Create the environment, install ATESTool, and run the example](#install-and-verify).

Use `ls` to list the current directory.

## macOS

1. If needed, [install Conda for macOS](https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html)
   and open a new Terminal window.
2. Run `git --version`. If Git is missing, run `xcode-select --install`, then repeat the check.
3. [Create the environment, install ATESTool, and run the example](#install-and-verify).

Use `ls` to list the current directory.

## Install and verify

**1. Create and activate the Python environment.** Enter one command at a time.
If Conda asks whether to proceed, enter `y`. Check that the last command prints `Python 3.12.x` before continuing.

```console
conda create -n atestool python=3.12 pip
conda activate atestool
python --version
```

**2. Clone the repository.** Run these commands in the folder where you want to
keep ATESTool. Use `mkdir models` to create a folder, `cd models` to enter it,
and `cd ..` to go up one level. Press Up Arrow to recall a previous command.

```console
git clone https://github.com/WouterDls/ATESTool-demo.git
cd ATESTool-demo
```

**3. Install ATESTool with the mesh backend.**

```console
python -m pip install ".[mesh]"
ATESTool --help
```

The `mesh` extra installs [Triangle](https://www.cs.cmu.edu/~quake/triangle.html)
for the refined meshes used in the exercises. Triangle is free for private,
research, and instructional use. Review its licence before commercial use.

**4. Install and check MODFLOW 6.** The examples use version 6.6.1. The
downloader installed with FloPy puts `mf6` in the active environment.

```console
get-modflow --repo modflow6 --release-id 6.6.1 :python
mf6 --version
```

**5. Run the first example.** This simulates five years and shows progress in
the terminal while MODFLOW runs.

```console
ATESTool examples/single_run.yaml --workspace examples/output/e1_single_run --report
```

Open `examples/output/e1_single_run/report.html` in a browser. In a new terminal,
run `conda activate atestool` and return to the repository folder before
running another example.

## Appendix: alternative installation options

- **Existing Python 3.12.** Follow the [Python 3.12 setup guide](https://docs.python.org/3.12/using/index.html)
  if you need the interpreter. From the repository folder, create a virtual
  environment with `py -3.12 -m venv .venv` on Windows or
  `python3.12 -m venv .venv` on Linux or macOS. Activate it with
  `.venv\Scripts\activate.bat` in Windows Command Prompt or
  `source .venv/bin/activate` in Bash or Zsh. Check `python --version`, then
  resume at step 3.
- **Repository ZIP.** On GitHub, choose **Code → Download ZIP**. Extract it and
  use `cd` to enter the extracted folder in place of step 2.
- **Editable and developer installs.** From the repository folder, use
  `python -m pip install -e ".[mesh]"` to work directly with the source.
  Use `python -m pip install -e ".[mesh,dev,docs]"` to add the type checker
  and documentation tools.
- **Installation without Triangle.** Use `python -m pip install .` to use the
  SciPy fallback. Its meshes lack the per-well refinement used in the exercises.
- **Manual MODFLOW installation.** Download the build for your operating system
  and CPU from the [MODFLOW 6.6.1 release page](https://github.com/MODFLOW-ORG/modflow6/releases/tag/6.6.1).
  Add its binary directory to `PATH` or set `MF6_BIN` to the full executable
  path. If using `MF6_BIN`, run that executable with `--version` to check it.
  ATESTool requires MODFLOW newer than 6.5.0.
