# Deprecated Tests

**Status:** This directory contains old test files that have been superseded by the reorganized test structure.

**Date:** October 2025

## What Happened

These tests have been refactored and migrated to the `tests/` directory following a clean architecture with proper separation of concerns:

- **Unit tests** → `tests/unit/`
- **Integration tests** → `tests/integration/`
- **End-to-end tests** → `tests/e2e/`
- **MCP tests** → `tests/mcp/`

## Migration Map

| Old Test File | New Location | Notes |
|---------------|--------------|-------|
| `test_correctness_validator.py` | `tests/unit/test_validators.py` | Merged with other validators |
| `test_cuda_backend.py` | `tests/unit/test_backends.py` | Consolidated with other backends |
| `test_io_contract.py` | `tests/unit/test_io_contract.py` | Direct migration |
| `test_server_*.py` | `tests/integration/test_endpoints.py`, `test_kernels.py` | Split by functionality |
| `test_triton_kernel.py` | `tests/integration/test_triton_metadata.py`, `tests/unit/test_backends.py` | Split between unit and integration |
| `test_server_comprehensive.py` | `tests/e2e/test_workflows.py` | Refactored for e2e testing |
| Other test files | Various locations in `tests/` | See tests/README.md for details |

## Why This Directory Exists

We're keeping these files temporarily to:
1. Ensure no test coverage was lost during migration
2. Provide reference for any edge cases
3. Allow gradual deprecation

## What To Do

- **For new tests:** Use `tests/` directory with the organized structure
- **For updates:** Find the equivalent test in `tests/` and update there
- **For debugging:** These old tests can serve as reference, but prefer the new structure

## Deletion Timeline

This directory will be archived/removed once:
- [ ] All functionality verified in new test structure
- [ ] Test coverage confirmed equivalent or better
- [ ] No references to old tests remain in documentation
- [ ] Team has been notified and transitioned

## Questions?

See `tests/README.md` for the new test organization or contact the maintainer.
