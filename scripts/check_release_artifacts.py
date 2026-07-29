#!/usr/bin/env python3
"""Validate the distributions in dist/ before they are uploaded to PyPI.

PyPI rejects an upload for reasons that a successful `python -m build` does not
catch (bad platform tag, version already taken, metadata mismatch). Twine only
uploads after building, so a rejection wastes the whole release run and, worse,
can leave a half-published release when some files upload and others do not.

This script runs the checks locally so a broken release is caught in CI instead.

Usage:
    python scripts/check_release_artifacts.py [--dist-dir dist] [--check-pypi]

Exit code 0 means the artifacts look publishable.
"""

import argparse
import json
import re
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPI_NAME = "daisy-exact-search"

# Platform tags PyPI accepts for a binary wheel. A wheel built by a plain
# `python -m build` on Linux is tagged `linux_x86_64`, which PyPI refuses:
# it carries no glibc floor, so pip cannot tell whether it will run.
ACCEPTED_PLATFORM_TAGS = ("any", "manylinux", "musllinux", "macosx", "win")

errors = []
warnings = []


def fail(msg):
    errors.append(msg)
    print("FAIL: " + msg)


def warn(msg):
    warnings.append(msg)
    print("WARN: " + msg)


def ok(msg):
    print("ok:   " + msg)


def read_declared_versions():
    """Return the version declared in each place the project records one."""
    versions = {}

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    if match:
        versions["pyproject.toml"] = match.group(1)

    setup_py = (PROJECT_ROOT / "setup.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', setup_py, re.MULTILINE)
    if match:
        versions["setup.py"] = match.group(1)

    init_py = (PROJECT_ROOT / "daisy" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_py, re.MULTILINE)
    if match:
        versions["daisy/__init__.py"] = match.group(1)

    return versions


def check_versions():
    """setup.py and pyproject.toml both feed the build, so they must agree."""
    versions = read_declared_versions()

    for source in ("pyproject.toml", "setup.py"):
        if source not in versions:
            fail("could not find a version declaration in " + source)

    build_versions = {k: v for k, v in versions.items() if k != "daisy/__init__.py"}
    if len(set(build_versions.values())) > 1:
        fail(
            "version mismatch between build files: "
            + ", ".join(k + "=" + v for k, v in build_versions.items())
        )
    else:
        ok("build files agree on version " + next(iter(build_versions.values())))

    # Not a publish blocker, but users read daisy.__version__ at runtime.
    runtime_version = versions.get("daisy/__init__.py")
    if runtime_version and runtime_version not in build_versions.values():
        warn(
            "daisy/__init__.py declares __version__ = "
            + runtime_version
            + " but the package is built as "
            + next(iter(build_versions.values()))
        )

    return next(iter(build_versions.values()))


def check_artifacts_present(dist_dir, version):
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))

    if not sdists:
        fail("no sdist (*.tar.gz) in " + str(dist_dir))
    elif len(sdists) > 1:
        fail("more than one sdist in " + str(dist_dir) + "; clean it before building")
    else:
        ok("found sdist " + sdists[0].name)

    if not wheels:
        fail("no wheel (*.whl) in " + str(dist_dir))
    else:
        ok("found " + str(len(wheels)) + " wheel(s): " + ", ".join(w.name for w in wheels))

    for artifact in sdists + wheels:
        # Wheel and sdist filenames normalise '-' to '_' in the project name.
        if version.replace("-", "_") not in artifact.name:
            fail(artifact.name + " does not carry the declared version " + version)

    return sdists, wheels


def check_wheel_tags(wheels):
    """Reject wheels PyPI will refuse, before we spend a release run finding out."""
    for wheel in wheels:
        parts = wheel.stem.split("-")
        if len(parts) < 5:
            fail(wheel.name + " is not a valid wheel filename")
            continue

        platform_tag = parts[-1]
        if platform_tag.startswith(ACCEPTED_PLATFORM_TAGS):
            ok(wheel.name + " has a PyPI-acceptable platform tag: " + platform_tag)
        else:
            fail(
                wheel.name
                + " has platform tag '"
                + platform_tag
                + "', which PyPI rejects. Run the wheel through `auditwheel repair` "
                  "(or build it with cibuildwheel) to get a manylinux tag."
            )


def check_sdist_contents(sdist):
    """The sdist is what users on unsupported platforms compile from."""
    required = [
        "pyproject.toml",
        "setup.py",
        "README.md",
        "daisy/__init__.py",
        "pybinds/setup.cpp",
    ]

    with tarfile.open(sdist, "r:gz") as tar:
        # Every path is prefixed with the '<name>-<version>/' root directory.
        names = {name.split("/", 1)[-1] for name in tar.getnames()}

    missing = [path for path in required if path not in names]
    if missing:
        fail(sdist.name + " is missing: " + ", ".join(missing))
    else:
        ok(sdist.name + " contains all required files")

    # setup.py compiles these unconditionally; if MANIFEST.in drops one, the
    # sdist builds here but fails for anyone installing from source.
    sources = [n for n in names if n.startswith(("lib/", "commons/")) and n.endswith((".cpp", ".hpp"))]
    if not sources:
        fail(sdist.name + " contains no C++ sources; check MANIFEST.in")
    else:
        ok(sdist.name + " contains " + str(len(sources)) + " C++ source/header files")


def check_pypi(version):
    """Warn if this version is already on PyPI; re-uploading is a hard error."""
    url = "https://pypi.org/pypi/" + PYPI_NAME + "/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            ok(PYPI_NAME + " is not on PyPI yet; any version can be published")
            return
        warn("could not query PyPI (HTTP " + str(exc.code) + "); skipping duplicate check")
        return
    except Exception as exc:
        warn("could not query PyPI (" + str(exc) + "); skipping duplicate check")
        return

    published = set(data.get("releases", {}))
    if version in published:
        warn(
            "version "
            + version
            + " is already published; the release job will skip the upload. "
              "Bump the version to publish."
        )
    else:
        ok("version " + version + " is not on PyPI yet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", default="dist", help="directory holding the built distributions")
    parser.add_argument("--check-pypi", action="store_true", help="also query PyPI for the version")
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir).resolve()
    if not dist_dir.is_dir():
        print("FAIL: " + str(dist_dir) + " does not exist; run `python -m build` first")
        return 1

    print("Checking release artifacts in " + str(dist_dir))
    print()

    version = check_versions()
    sdists, wheels = check_artifacts_present(dist_dir, version)
    check_wheel_tags(wheels)
    for sdist in sdists:
        check_sdist_contents(sdist)
    if args.check_pypi:
        check_pypi(version)

    print()
    if errors:
        print(str(len(errors)) + " check(s) failed; these artifacts are not safe to upload.")
        return 1

    print("All checks passed" + (" (" + str(len(warnings)) + " warning(s))" if warnings else "") + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
