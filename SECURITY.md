# Security

Do not place passwords, tokens, private SSH configuration, machine names,
addresses, model weights, or raw host logs in an issue or pull request.

For a security problem in this recipe, use GitHub's private security-advisory
reporting for this repository. For a problem in TensorFold, MCDMA, Mia's
recipe, oMLX, a model, or a container, report it to that upstream project.

Before publishing a change, run:

```sh
python3 scripts/check-public-release.py
```

Raw benchmark evidence belongs under ignored `results/raw/`; only a reviewed,
sanitized receipt belongs in Git.
