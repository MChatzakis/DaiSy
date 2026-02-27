#!/bin/bash

# PyPI Publishing Script
# Usage: ./publish_pip.sh <version>
# Example: ./publish_pip.sh 1.0.6

if [ -z "$1" ]; then
    echo "Error: Version number required"
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.6"
    exit 1
fi

VERSION=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Publishing version $VERSION to PyPI..."
echo

# Update version in setup.py
echo "Updating setup.py..."
sed -i "s/__version__ = \"[^\"]*\"/__version__ = \"$VERSION\"/" "$PROJECT_ROOT/setup.py"

# Update version in pyproject.toml
echo "Updating pyproject.toml..."
sed -i "s/^version = \"[^\"]*\"/version = \"$VERSION\"/" "$PROJECT_ROOT/pyproject.toml"

# Clean previous builds
echo "Cleaning previous builds..."
cd "$PROJECT_ROOT"
rm -rf build dist *.egg-info

# Build the package
echo "Building package..."
python -m build

if [ $? -ne 0 ]; then
    echo "Error: Build failed"
    exit 1
fi

echo
# Upload to PyPI
echo "Uploading to PyPI..."
python -m twine upload dist/*

if [ $? -eq 0 ]; then
    echo
    echo "✓ Successfully published version $VERSION to PyPI"
else
    echo "Error: Upload failed"
    exit 1
fi
