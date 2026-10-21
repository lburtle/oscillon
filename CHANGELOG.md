# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project's version is derived from git tags via `hatch-vcs`.

## [Unreleased]

### Changed

- Reorganized `src/oscillon` into subpackages: `ctln` (weight construction,
  simulation, dynamics, readout), `search` (OpenAI-ES optimizer, fitness,
  training loop), plus placeholder `analysis` and `io` subpackages for
  future fixed-point/oscillation-metric and config/result-serialization
  code. Added an optional `oscillon` CLI entry point (`oscillon.cli:main`).
- Moved the accumulated research scripts, sweep logs, and result dumps
  from `examples/` into `experiments/` (paper-specific, not packaged).
  Replaced `examples/` with a couple of short, runnable scripts that
  import `oscillon` and demonstrate basic usage.

## [0.1.0] - Unreleased

- Initial gCTLN model: soft/block topology specs, forward-Euler dynamics
  simulation, closed-form readout fitting, and OpenAI-ES-based topology
  search.
