# Release workflow

0.3.0 was the first PyPI release; 0.4.0 adds adapters and comparison reports.
Do not publish 0.2.0.
Public Git history preserves the earlier defect and its correction.

## Trusted Publishing setup

On PyPI, verify the account email and complete required two-factor setup.
Under account Publishing, add a pending GitHub publisher for the first release:

- PyPI project: `calibroute-ai`
- GitHub owner: `Garyouki`
- Repository: `calibroute`
- Workflow filename: `publish.yml`
- Environment: `pypi`

The workflow `.github/workflows/publish.yml` is manually dispatched from `main`.
It tests and builds the selected commit, then publishes the resulting artifacts
in a separate job with short-lived OIDC credentials. No password or API token
belongs in this repository. Configure environment protection on GitHub if
additional release approval is desired. Do not dispatch until the corresponding
PyPI publisher has been configured and release metadata is finalized.

## Release checks

1. Run tests, lint, the case study, and build checks below.
2. Check package version, `__version__`, CITATION, and changelog agree.
3. On actual release, set CITATION `date-released` to that release date and
   replace the changelog's pending marker. Create an annotated version tag
   and a GitHub Release pointing at the verified commit.
4. Publish built artifacts using PyPI Trusted Publishing or a token configured
   locally. Never put a token in chat, source, or a committed `.pypirc`.
5. Verify installation from PyPI in a fresh environment before describing the
   package as published. Add the PyPI install command as the default README path.

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
ruff check src tests examples/financial_ner/run_case_study.py
ruff format --check src tests examples/financial_ner/run_case_study.py
python examples/financial_ner/run_case_study.py
python -m build
python -m twine check dist/*
```

Preserve the release URL, commit, checksums, CI result, and reproducible outputs.
For an impact record, distinguish authored software from independent adoption:
record only verifiable external uses, citations, contributed issues, and feedback
with their dates and links. A repository or passing CI alone does not establish
independent use or practical impact.
