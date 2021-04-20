## Contributing

We welcome contributions to this repository. When contributing to this repository, we encourage you to discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Code contributions should be made via a github pull request (PR).
Once a PR has been submitted, we conduct a review, so please anticipate some discussion on anything you submit to ensure we can keep the code base of a good quality and maintainable.

### Layout

This repository is ess instrument centric and we therefore strongly prefer having all code organised into `src/ess/{instrument}` directories.
If your code is specific to the ess facility, but is intended to be ubiquitous across a class of instruments, you may put it into a technique specific directory `src/ess/{technique}` i.e `src/ess/reflectometry`.
Code that is technique specific, but free from ess facility considerations may be considered to go in [scippneutron](https://github.com/scipp/scippneutron)

### Testing

To ensure that the package is robust we are very keen that authors provide unit tests alongside code.
It is possible that future updates for `scipp`, `scippneutron` dependencies can break the code you contribute.
If we are aware of failing tests, we can provide future fixes and migrations for you. Please avoid large data files, or any code requiring network access.
Test suites should be fast to execute.

## Documentation

Please provide and update documentation.
Put python docstrings on your user facing functions, provide code comments and consider other explanations you need to include to describe how your functions work.
We will build and publish sphinx documentation located [here](https://github.com/scipp/ess/tree/main/docs).
