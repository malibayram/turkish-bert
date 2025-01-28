# Contributing to Turkish BERT

First off, thank you for considering contributing to Turkish BERT! It's people like you that make Turkish BERT such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include details about your configuration and environment

### Suggesting Enhancements

If you have a suggestion for the project, we'd love to hear it. Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Include screenshots and animated GIFs in your pull request whenever possible
* Follow the Python style guide
* Include tests when adding new features
* Update documentation when changing functionality

## Development Process

1. Fork the repo
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Local Development

```bash
# Clone your fork
git clone https://github.com/your-username/turkish-bert.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest
```

### Code Style

* Follow PEP 8 guidelines
* Use meaningful variable names
* Add comments for complex logic
* Keep functions focused and small
* Write docstrings for functions and classes

### Testing

* Write unit tests for new features
* Ensure all tests pass before submitting PR
* Include both positive and negative test cases
* Test edge cases

## Project Structure

```
.
├── bert/                  # Core BERT implementation
├── turkish_tokenizer/     # Tokenizer implementations
├── tests/                 # Test files
└── docs/                  # Documentation
```

## Documentation

* Keep README.md up to date
* Document new features
* Update docstrings
* Include examples in documentation

## Questions?

Feel free to contact the project team at [GitHub Issues](https://github.com/malibayram/turkish-bert/issues).

## License

By contributing, you agree that your contributions will be licensed under its MIT License. 