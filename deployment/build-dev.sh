#!/bin/bash
# =============================================================================
# Build & Dev Script - isA_Agent_SDK
# =============================================================================
# Usage:
#   ./deployment/build-dev.sh --setup     Create venv + install deps (editable)
#   ./deployment/build-dev.sh --test      Run tests with dev.env loaded
#   ./deployment/build-dev.sh --build     Build sdist + wheel into dist/
#   ./deployment/build-dev.sh --publish   Upload dist/ to PyPI (twine)
#   ./deployment/build-dev.sh --status    Show installed ISA packages
#   ./deployment/build-dev.sh             Full setup (default)
# =============================================================================
set -e

PROJECT_NAME="isa_agent_sdk"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

case "${1:-}" in
    --setup)
        echo "Setting up $PROJECT_NAME..."

        # Create venv
        rm -rf .venv
        uv venv .venv --python 3.12

        # Install ISA platform packages (editable) + dev tools
        uv pip install -r deployment/requirements/dev.txt --python .venv/bin/python

        # Install the SDK itself in editable mode with dev extras
        uv pip install -e ".[dev]" --python .venv/bin/python

        echo ""
        echo "Setup complete!"
        echo "  Activate:  source .venv/bin/activate"
        echo "  Test:      $0 --test"
        echo "  Build:     $0 --build"
        ;;
    --test)
        echo "Running $PROJECT_NAME tests..."

        # Load dev environment
        set -a
        source deployment/environments/dev.env 2>/dev/null || true
        set +a

        shift
        .venv/bin/python -m pytest tests/ -v "$@"
        ;;
    --build)
        echo "Building $PROJECT_NAME package..."

        # Clean previous builds
        rm -rf build dist *.egg-info

        # Build sdist + wheel
        .venv/bin/python -m build

        echo ""
        echo "Build artifacts:"
        ls -lh dist/
        ;;
    --publish)
        echo "Publishing $PROJECT_NAME to PyPI..."

        if [ ! -d dist ] || [ -z "$(ls dist/ 2>/dev/null)" ]; then
            echo "No dist/ artifacts found. Run --build first."
            exit 1
        fi

        # Ensure twine is available
        uv pip install twine --python .venv/bin/python

        .venv/bin/twine upload dist/*
        ;;
    --status)
        echo "Venv: .venv"
        uv pip show isa-common isa-mcp isa-model isa-agent-sdk --python .venv/bin/python 2>/dev/null \
            | grep -E "Name|Version|Location" || echo "Not installed"
        ;;
    *)
        # Default: full setup
        $0 --setup
        ;;
esac
