# AGENTS.md

SpikeSpy is a PySide6 desktop app for annotating stimulus-locked microneurography recordings. The data model is `neo`/`quantities`; the native project format is NIX `.h5`.

## Setup and commands

- Python is constrained to `>=3.10,<3.13` (CI and `.venv` use 3.12). Use `uv`, not pip/poetry.
- `uv sync` installs the app plus the `dev` dependency group from `uv.lock`.
- `uv run spikespy [path/to/file.h5]` starts the GUI (entry point `spikespy.SpikeSpy:run`). `uv run python run.py [file]` is the PyInstaller entry.
- Headless test run: `$env:QT_QPA_PLATFORM="offscreen"; uv run pytest tests -q`. Omit the env prefix only when real windows are wanted.
- Single test: `uv run pytest tests/test_channel_group_container.py::test_channel_group_container_basics -q`
- `uv run black <files>` is available, but Black is the only configured formatter; there is no lint, typecheck, or pre-commit config. Avoid mass-formatting the repo.
- Build the Windows executable from the repo root: `uv run pyinstaller SpikeSpy.spec` (outputs `dist/SpikeSpy.exe`). CI (`.github/workflows/build.windows.yml`) runs only on `v*` tags and uploads `dist/SpikeSpy.exe` to the GitHub release for that tag; the release must already exist. Version comes from `pyproject.toml`.

## Testing gotchas

- Tests use pytest + pytest-qt. Fixtures are in `tests/conftest.py` and per-file; the autouse fixture sets `ViewerState.live_frame_interval_ms = 0` so live updates are emitted synchronously. Tests may rely on that.
- Live-stream tests start `tests/mock_open_ephys_server.py` (ZMQ data/event publishers plus an HTTP control server) on fixed localhost ports. Run these serially; parallel test runs can collide on ports.
- `data/` is gitignored. Tests that need `data/test2.h5` skip when it is absent.
- `tests/test_APTrackConfig.py` is a manual `__main__` script, not a pytest test (it collects zero tests).
- Some suites assert frame/timing behavior; prefer focused test files while iterating.

## Architecture

- `spikespy/SpikeSpy.py` is the composition root: `run()` builds `MdiView`, and `window_options` is the panel registry. Register new panels there; views implement `get_settings()`/`set_settings()` to persist their settings in layout perspectives.
- `ViewerState` is the shared `QObject` model behind all panels (segment, analog/event signals, `tracked_neuron_unit` spike groups, live buffers). Views observe it rather than owning data.
- Most panels have Matplotlib and PyQtGraph twins (`FooView.py` / `FooPGView.py`); behavior changes often need both. PyQtGraph views are imported optionally in `SpikeSpy.py` (`_has_pg`), so keep import failures non-fatal.
- Channel selection is handled by `ChannelGroupBar` and `ChannelGroupContainer`; `MdiView.newWindow` auto-wraps channel/unit-specific views when given a unit or channel target. Views integrate by exposing a `channel_bar`.
- Live acquisition lives in `open_ephys_addin/` (ZMQ stream worker, `OpenEphysController` HTTP control, singleton `OpenEphysLiveView`). DSP primitives are in `processing.py` (`LiveERPBuffer`, `MultiAnalogSignal`, `OnlineBandpassFilter`, `OnlineTTLDetector`, `MultiChannelTemplateMatcher`). Auto-tracking is in `basic_tracking.py` + `tracking_worker.py`, and render offloading is in `viewer_worker.py`. Keep expensive work off the GUI thread and honor live frame throttling.
- I/O is centralized in `io_manager.py`: loads NIX `.h5`, NWB, Open Ephys binary, MATLAB folders, APTrack, and Spike2; saves NIX/NWB; imports/exports unit CSV and NPZ. `NeoOpenEphyisIO.py` (note the misspelling) is the legacy custom Open Ephys reader; its module name is imported as-is.
- `oe_plugin/` is a separate C++ Open Ephys GUI plugin (CMake + libzmq). `uv` does not build or package it; CI builds it via `.github/workflows/build.oe_plugin.yml` on `v*` tags, PRs touching `oe_plugin/**`, and manual dispatch (ZeroMQ comes from vcpkg on Windows, Homebrew on macOS, `libzmq3-dev` on Linux).

## Repo hygiene

- Do not edit `.kilo/worktrees/**` from the main workspace; those are Agent Manager git worktrees containing full, potentially stale copies.
- `.venv/`, `data/`, `build/`, and `dist/` are ignored; do not commit generated artifacts.
