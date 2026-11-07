# Security policy

## Reporting a vulnerability

Please report security problems privately through GitHub's private vulnerability reporting. On the repository page, open the **Security and quality** tab and choose **Report a vulnerability**, or go straight to the [advisory form](https://github.com/mertkayacs/jevalt/security/advisories/new). Please do not open a public issue for a security problem.

Include the affected file or command, the steps to reproduce it and the impact you expect. I read every report and reply in the private advisory.

## Scope

This policy covers the code in this repository: the `jevalt` package with its server and command line, and the training and export scripts in `training/`. The server binds to `127.0.0.1` by default. If you expose it, set `JEVALT_API_KEY` as described in [Run locally](docs/local.md#security).

A model that follows a planted instruction or picks a wrong option is a known limit of the released models, measured in the README, and is out of scope here. Please report those with the **Model result question** issue form.

## Supported versions

Fixes go to the `main` branch. The repository has no release branches.
