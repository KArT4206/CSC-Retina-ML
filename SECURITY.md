# Security policy

## Scope

This policy covers the CSC OCT Analyser web application and the research pipeline. It does not cover Hugging Face's hosting infrastructure or the public datasets.

## Data handling

When `CSC_PUBLIC=1` (as on the live demo), uploaded scans are deleted after analysis and the history page shows nothing, so one visitor cannot see another's scans. Do not upload identifiable patient images to the public demo.

## Reporting a vulnerability

Please report security issues privately to the repository owner instead of opening a public issue. Include a description, steps to reproduce and any relevant screenshots. Reports are handled on a best-effort basis; this is a research project with no dedicated security team.
