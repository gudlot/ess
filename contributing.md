## Contributing

We welcome contributions to this repository. When contributing to this repository, we encourage you to discuss the change you wish to make via issue, email, or any other method with the owners of this repository before making a change.

Code contributions should be made via a github pull request (PR). Once a PR has been submitted, we conduct a review, so please anticipate some discussion on anything you submit to ensure we can keep the code base of a good quality and maintainable.

### Layout

This repository is ess instrument centric and we therefore strongly prefer having all code organised into `src/ess/{instrument}` directories. Please keep all code in this structure even if the code is in principle generic for a class of instruments. Code that is robust and generic across and instrument suite may be included in the dependency codebase [scippneutron](https://github.com/scipp/scippneutron)

### Testing

To ensure that the package is robust we are very keen that authors provide unit tests alongside code. It is possible that future updates for `scipp`, `scippneutron` dependencies can break the code you contribute. If we are aware of failing tests, we can provide future fixes and migrations for you. Please avoid large data files, or any code requiring network access. Test suites should be fast to execute.
