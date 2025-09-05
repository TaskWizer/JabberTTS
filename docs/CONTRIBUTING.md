# Contributing to JabberTTS

Thank you for your interest in contributing to JabberTTS! This document provides guidelines and information for contributors to help maintain code quality and project consistency.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Development Workflow](#development-workflow)
5. [Code Standards](#code-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Issue Reporting](#issue-reporting)
10. [Performance Guidelines](#performance-guidelines)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. By participating, you agree to:

- Be respectful and inclusive in all interactions
- Focus on constructive feedback and collaboration
- Help maintain a positive community environment
- Report any unacceptable behavior to project maintainers

## Getting Started

### Prerequisites
- Python 3.9+ (recommended: 3.11)
- Git
- Docker (for containerized development)
- uv package manager

### First-Time Setup
```bash
# Clone the repository
git clone https://github.com/TaskWizer/JabberTTS.git
cd JabberTTS

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup development environment
uv sync --dev

# Verify installation
uv run python -c "import jabbertts; print('Setup successful!')"
```

## Development Setup

### Environment Configuration
```bash
# Create development environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies with development tools
uv sync --dev

# Install pre-commit hooks
uv run pre-commit install
```

### Docker Development
```bash
# Build development container
docker build -t jabbertts-dev .

# Run development server
docker run -p 8000:8000 -v $(pwd):/app jabbertts-dev
```

### IDE Configuration
Recommended VS Code extensions:
- Python
- Black Formatter
- Pylance
- GitLens
- Docker

## Development Workflow

### Branch Strategy
- **main**: Stable, production-ready code
- **develop**: Integration branch for features
- **feature/**: New features (`feature/voice-cloning`)
- **bugfix/**: Bug fixes (`bugfix/memory-leak`)
- **hotfix/**: Critical production fixes

### Workflow Steps
1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Test**
   ```bash
   # Make changes
   # Run tests frequently
   uv run pytest
   
   # Check code quality
   uv run black .
   uv run flake8
   uv run mypy jabbertts/
   ```

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add voice cloning endpoint"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create pull request on GitHub
   ```

### Commit Message Format
Follow conventional commits format:
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/modifications
- `chore`: Maintenance tasks

**Examples:**
```
feat(api): add voice cloning endpoint
fix(inference): resolve memory leak in model loading
docs(readme): update installation instructions
test(api): add integration tests for speech endpoint
```

## Code Standards

### Python Style Guide
- Follow PEP 8 style guidelines
- Use Black for code formatting
- Maximum line length: 88 characters
- Use type hints for all functions and methods

### Code Quality Tools
```bash
# Format code
uv run black .

# Check style
uv run flake8

# Type checking
uv run mypy jabbertts/

# Import sorting
uv run isort .

# Security linting
uv run bandit -r jabbertts/
```

### Code Structure
```python
"""Module docstring describing purpose and usage."""

import standard_library
import third_party_library

from jabbertts import local_module

# Constants
DEFAULT_SAMPLE_RATE = 24000

class ExampleClass:
    """Class docstring with purpose and usage examples."""
    
    def __init__(self, param: str) -> None:
        """Initialize with clear parameter documentation."""
        self.param = param
    
    def public_method(self, input_data: str) -> str:
        """Public method with type hints and docstring."""
        return self._private_method(input_data)
    
    def _private_method(self, data: str) -> str:
        """Private method for internal use."""
        return data.upper()
```

### Error Handling
```python
# Use specific exceptions
class JabberTTSError(Exception):
    """Base exception for JabberTTS errors."""
    pass

class ModelLoadError(JabberTTSError):
    """Raised when model loading fails."""
    pass

# Proper error handling
try:
    result = risky_operation()
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise JabberTTSError(f"Failed to process: {e}") from e
```

## Testing Guidelines

### Test Structure
```
tests/
├── unit/           # Unit tests for individual functions
├── integration/    # Integration tests for components
├── performance/    # Performance and benchmark tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests
```python
import pytest
from jabbertts.inference import InferenceEngine

class TestInferenceEngine:
    """Test suite for InferenceEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create test inference engine."""
        return InferenceEngine(model_path="test_model.onnx")
    
    def test_generate_speech_success(self, engine):
        """Test successful speech generation."""
        text = "Hello, world!"
        audio = engine.generate(text)
        
        assert audio is not None
        assert len(audio) > 0
        assert isinstance(audio, bytes)
    
    def test_generate_speech_empty_text(self, engine):
        """Test error handling for empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            engine.generate("")
    
    @pytest.mark.performance
    def test_generation_performance(self, engine):
        """Test performance requirements."""
        text = "Performance test sentence."
        start_time = time.time()
        audio = engine.generate(text)
        duration = time.time() - start_time
        
        # RTF should be < 0.5
        audio_duration = len(audio) / 24000  # Assuming 24kHz
        rtf = duration / audio_duration
        assert rtf < 0.5, f"RTF {rtf} exceeds target"
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/unit/
uv run pytest tests/integration/
uv run pytest -m performance

# Run with coverage
uv run pytest --cov=jabbertts --cov-report=html

# Run tests in parallel
uv run pytest -n auto
```

### Test Requirements
- **Unit Tests**: 80% minimum coverage
- **Integration Tests**: All API endpoints
- **Performance Tests**: RTF and memory validation
- **Quality Tests**: Audio quality metrics

## Documentation

### Code Documentation
- Use clear, descriptive docstrings
- Include usage examples in docstrings
- Document all parameters and return values
- Add type hints for all functions

### API Documentation
- OpenAPI/Swagger specifications
- Request/response examples
- Error code documentation
- Authentication details

### User Documentation
- Installation guides
- Usage examples
- Configuration reference
- Troubleshooting guides

### Documentation Format
```python
def generate_speech(
    text: str,
    voice: str = "alloy",
    speed: float = 1.0
) -> bytes:
    """Generate speech audio from input text.
    
    Args:
        text: Input text to convert to speech (max 4096 chars)
        voice: Voice identifier (alloy, echo, fable, onyx, nova, shimmer)
        speed: Speech speed multiplier (0.25-4.0)
    
    Returns:
        Audio data in MP3 format
    
    Raises:
        ValueError: If text is empty or too long
        ModelError: If speech generation fails
    
    Example:
        >>> audio = generate_speech("Hello, world!", voice="alloy")
        >>> with open("output.mp3", "wb") as f:
        ...     f.write(audio)
    """
```

## Pull Request Process

### Before Submitting
1. **Run Full Test Suite**
   ```bash
   uv run pytest
   uv run pytest --cov=jabbertts
   ```

2. **Check Code Quality**
   ```bash
   uv run black --check .
   uv run flake8
   uv run mypy jabbertts/
   ```

3. **Update Documentation**
   - Update relevant docstrings
   - Add/update user documentation
   - Update CHANGELOG.md

4. **Performance Validation**
   ```bash
   uv run pytest -m performance
   ```

### PR Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature causing existing functionality to change)
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] CHANGELOG.md updated
```

### Review Process
1. **Automated Checks**: CI/CD pipeline validation
2. **Code Review**: Maintainer review for quality and design
3. **Testing**: Comprehensive test validation
4. **Performance**: Performance impact assessment
5. **Documentation**: Documentation completeness check

## Issue Reporting

### Bug Reports
Use the bug report template:
```markdown
**Bug Description**
Clear description of the bug.

**Reproduction Steps**
1. Step one
2. Step two
3. Step three

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.11.5]
- JabberTTS: [e.g., 0.2.0]

**Additional Context**
Any other relevant information.
```

### Feature Requests
Use the feature request template:
```markdown
**Feature Description**
Clear description of the proposed feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches considered.

**Additional Context**
Any other relevant information.
```

## Performance Guidelines

### Performance Targets
- **RTF (Real-Time Factor)**: < 0.5
- **Memory Usage**: < 2GB
- **First Token Latency**: < 1s
- **Concurrent Requests**: 100+

### Performance Testing
```python
@pytest.mark.performance
def test_rtf_performance():
    """Test RTF performance requirement."""
    engine = InferenceEngine()
    text = "Performance test with moderate length text."
    
    start_time = time.time()
    audio = engine.generate(text)
    inference_time = time.time() - start_time
    
    audio_duration = len(audio) / 24000
    rtf = inference_time / audio_duration
    
    assert rtf < 0.5, f"RTF {rtf} exceeds target of 0.5"
```

### Optimization Guidelines
- Profile code before optimizing
- Focus on critical path optimization
- Use appropriate data structures
- Minimize memory allocations
- Cache expensive computations

## Getting Help

### Community Support
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and community support
- **Discord**: Real-time community chat (coming soon)

### Maintainer Contact
- Create GitHub issue for bugs/features
- Tag maintainers for urgent issues
- Follow up on stale PRs after 1 week

### Resources
- [Project Documentation](../README.md)
- [API Reference](SPEC.md)
- [Performance Guide](PLAN.md)
- [Testing Strategy](TESTING.md)

Thank you for contributing to JabberTTS! Your contributions help make high-quality TTS technology accessible to everyone.
