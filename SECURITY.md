# Security Policy

## Attack Surface

rhombic is a pure computation library. It imports numpy and networkx. It constructs graphs, computes metrics, and optionally renders plots. It does not:

- Connect to any network or external service
- Handle authentication or user credentials
- Process user input beyond function arguments
- Read or write files beyond optional plot saving

The security surface is effectively zero.

## Reporting

If you find a vulnerability (dependency issue, unexpected file access, etc.):

**Email:** security@promptcrafted.com

Allow 72 hours for initial response. Do not open public issues for security concerns.

## Dependencies

| Package | Purpose | Exposure |
|---------|---------|----------|
| numpy | Array operations | None (numerical computation) |
| networkx | Graph construction & metrics | None (in-memory graphs) |
| matplotlib | Optional plotting | File write (plot saving only) |

## Pre-commit Scanning

This repo uses gitleaks to prevent accidental secret commits. Install with:

```bash
pip install pre-commit
pre-commit install
```
