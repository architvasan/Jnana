# Jnana Installation Guide

This guide explains how to install Jnana as a Python package using the new `pyproject.toml` configuration.

---

## 📦 Installation Methods

### Method 1: Editable Install (Recommended for Development)

This method installs Jnana in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.

```bash
# Navigate to Jnana directory
cd /path/to/Jnana

# Install in editable mode
pip install -e .
```

**Benefits:**
- ✅ Changes to code are immediately available
- ✅ Perfect for active development
- ✅ Easy to test modifications

---

### Method 2: Editable Install with Development Tools

Install Jnana with all development dependencies (testing, linting, formatting):

```bash
cd /path/to/Jnana
pip install -e ".[dev]"
```

**Includes:**
- pytest, pytest-asyncio, pytest-mock, pytest-cov
- black, isort, mypy, ruff
- pre-commit hooks

---

### Method 3: Editable Install with All Optional Dependencies

Install everything (dev tools, docs, testing):

```bash
cd /path/to/Jnana
pip install -e ".[all]"
```

---

### Method 4: Standard Install (Production)

Install Jnana as a regular package (not editable):

```bash
cd /path/to/Jnana
pip install .
```

**Use when:**
- Deploying to production
- Installing on a server
- Don't need to modify source code

---

## 🎯 Quick Start for StructBioReasoner Integration

For your use case (integrating Jnana with StructBioReasoner), use **editable install**:

```bash
# 1. Install Jnana in editable mode
cd ~/Desktop/Code/Jnana
pip install -e .

# 2. Verify installation
python -c "from jnana.protognosis.core.coscientist import CoScientist; print('✅ Jnana installed!')"

# 3. Run your integration tests
cd ~/Desktop/Code/StructBioReasoner
python test_quick_integration.py
python test_jnana_structbioreasoner_integration.py
```

---

## 🔧 Installation Options

### Core Installation (Minimal)

```bash
pip install -e .
```

**Includes:**
- All core dependencies
- LLM integrations (OpenAI, Anthropic, Google, Ollama)
- Data processing (pandas, numpy)
- UI components (rich, textual)
- Biomni integration
- LangChain components

---

### Development Installation

```bash
pip install -e ".[dev]"
```

**Additional tools:**
- pytest (testing framework)
- black (code formatter)
- isort (import sorter)
- mypy (type checker)
- ruff (fast linter)
- pre-commit (git hooks)

---

### Documentation Installation

```bash
pip install -e ".[docs]"
```

**Additional tools:**
- sphinx (documentation generator)
- sphinx-rtd-theme (Read the Docs theme)
- myst-parser (Markdown support)

---

### Testing Installation

```bash
pip install -e ".[test]"
```

**Additional tools:**
- pytest and plugins
- pytest-asyncio (async test support)
- pytest-mock (mocking support)
- pytest-cov (coverage reporting)
- pytest-timeout (timeout support)

---

## ✅ Verify Installation

### Check Package Installation

```bash
# Check if jnana is installed
pip show jnana

# Expected output:
# Name: jnana
# Version: 0.1.0
# Summary: AI Co-Scientist with Interactive Hypothesis Generation
# Location: /path/to/Jnana
# Requires: pyyaml, pydantic, openai, anthropic, ...
```

---

### Test Imports

```bash
# Test basic imports
python -c "import jnana; print(f'Jnana version: {jnana.__version__}')"

# Test core components
python -c "from jnana.core.jnana_system import JnanaSystem; print('✅ JnanaSystem')"
python -c "from jnana.protognosis.core.coscientist import CoScientist; print('✅ CoScientist')"
python -c "from jnana.data.unified_hypothesis import UnifiedHypothesis; print('✅ UnifiedHypothesis')"
```

---

### Run Tests

```bash
# Navigate to Jnana directory
cd /path/to/Jnana

# Run all tests
pytest

# Run specific test
pytest tests/test_unified_hypothesis.py

# Run with coverage
pytest --cov=jnana --cov-report=html
```

---

## 🔄 Updating Installation

### After Pulling New Changes

If you installed in editable mode (`-e`), changes are automatically available. Just pull the latest code:

```bash
cd /path/to/Jnana
git pull origin main
```

No need to reinstall!

---

### After Changing Dependencies

If `pyproject.toml` dependencies change, reinstall:

```bash
cd /path/to/Jnana
pip install -e . --upgrade
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'jnana'"

**Solution:**
```bash
# Make sure you installed the package
cd /path/to/Jnana
pip install -e .

# Check if it's in your Python path
python -c "import sys; print('\n'.join(sys.path))"
```

---

### Issue: "ModuleNotFoundError: No module named 'biomni'"

**Solution:**
```bash
# Install biomni separately if needed
pip install biomni

# Or reinstall jnana
pip install -e . --force-reinstall
```

---

### Issue: "uvloop not available on Windows"

**Solution:**
This is expected! `uvloop` is only installed on non-Windows systems (see `pyproject.toml`):
```toml
"uvloop>=0.19.0; sys_platform != 'win32'"
```

On Windows, asyncio uses the default event loop.

---

### Issue: Dependency Conflicts

**Solution:**
```bash
# Create a fresh virtual environment
python -m venv venv_jnana
source venv_jnana/bin/activate  # On Windows: venv_jnana\Scripts\activate

# Install jnana
cd /path/to/Jnana
pip install -e .
```

---

## 📚 What Gets Installed

### Package Structure

```
jnana/
├── __init__.py              # Package initialization
├── core/                    # Core system components
│   ├── jnana_system.py
│   ├── model_manager.py
│   └── ...
├── protognosis/             # ProtoGnosis integration
│   ├── core/
│   │   └── coscientist.py
│   ├── agents/
│   │   └── specialized_agents.py
│   └── ...
├── data/                    # Data models
│   └── unified_hypothesis.py
├── agents/                  # Agent implementations
├── utils/                   # Utility functions
└── ui/                      # User interface components
```

---

### Entry Points

After installation, you can run:

```bash
# Command-line interface (if implemented)
jnana --help
```

---

## 🎯 Integration with StructBioReasoner

### Recommended Setup

```bash
# 1. Install Jnana in editable mode
cd ~/Desktop/Code/Jnana
pip install -e .

# 2. Install StructBioReasoner in editable mode
cd ~/Desktop/Code/StructBioReasoner
pip install -e .

# 3. Both packages are now available
python -c "import jnana; import struct_bio_reasoner; print('✅ Both installed!')"
```

---

### Update Your Test Files

You can now remove the manual path insertion:

**Before:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / '../Jnana'))  # ❌ Not needed anymore!

from jnana.protognosis.core.coscientist import CoScientist
```

**After:**
```python
from jnana.protognosis.core.coscientist import CoScientist  # ✅ Clean import!
```

---

## 📝 Configuration Files

### pyproject.toml Features

The new `pyproject.toml` includes:

1. **Build System**: Modern setuptools configuration
2. **Dependencies**: All required packages with version constraints
3. **Optional Dependencies**: Dev, docs, test, all
4. **Scripts**: Command-line entry points
5. **Tool Configuration**: black, isort, mypy, pytest, ruff

---

### Development Tools Configuration

All tools are pre-configured in `pyproject.toml`:

```bash
# Format code with black
black jnana/

# Sort imports with isort
isort jnana/

# Type check with mypy
mypy jnana/

# Lint with ruff
ruff check jnana/

# Run tests with pytest
pytest
```

---

## 🚀 Next Steps

After installation:

1. ✅ **Verify installation** with test imports
2. ✅ **Run integration tests** in StructBioReasoner
3. ✅ **Remove manual path insertions** from your code
4. ✅ **Use clean imports** everywhere

---

## 📞 Support

If you encounter issues:

1. Check this guide's troubleshooting section
2. Verify Python version: `python --version` (should be ≥3.9)
3. Check pip version: `pip --version` (should be recent)
4. Try creating a fresh virtual environment

---

**Installation complete! 🎉**

