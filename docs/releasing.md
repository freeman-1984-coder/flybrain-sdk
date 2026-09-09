# Release checklist

The repository name and package name are provisional. PyPI/NPM name availability
and account ownership must be checked before attempting registry publication.
This project currently supports installation from GitHub and release wheels.

1. Run CI, quickstart, `python -m build` and `python -m twine check dist/*`.
2. Install the built wheel in a clean environment, outside the checkout; verify the
   toy, registry catalog, and checkpoint roundtrip are included and usable.
3. Update version in `pyproject.toml` and `src/flybrain/__init__.py`; write release notes.
4. Create a version tag and GitHub release, attaching the wheel and source distribution.
5. Configure PyPI trusted publishing under the real package owner before publishing
   to PyPI. Start on TestPyPI if needed. Never put publishing tokens into this repo.

The TypeScript package remains private until a runtime exists. Do not publish
interface declarations as if they were a working WASM brain.
