# Release checklist

The repository name and package name are provisional. PyPI/NPM name availability
and account ownership must be checked before attempting registry publication.
This project currently supports installation from GitHub and release wheels.

1. Update version in `pyproject.toml` and `src/flybrain/__init__.py`; write release notes.
2. Run CI, quickstart, `python -m build` and `python -m twine check dist/*`.
3. Install the built wheel in a clean environment, outside the checkout; verify the
   toy, registry catalog, and checkpoint roundtrip are included and usable.
   Verify that the source archive includes referenced validation JSON/XML/CSV/text
   evidence, not only the Markdown reports. Check optional CUDA imports without CuPy.
4. Create a version tag and GitHub release, attaching the wheel and source distribution.
   After the tag is public, install a generated demo's `requirements.txt` in another
   clean environment and run it: project generation pins this exact public tag.
5. Configure PyPI trusted publishing under the real package owner before publishing
   to PyPI. Start on TestPyPI if needed. Never put publishing tokens into this repo.

The TypeScript workspace package has a tested JavaScript CPU runtime and remains
private pending its own packaging/registry release process. It is not a working
WASM backend; do not publish or describe it as one.
