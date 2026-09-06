"""Reproducible environment: pyproject, uv.lock, Dockerfile and CI must agree.

SDD task V01 (finding F14, requirement Q18).  The image used to be built
from ``pip install -e .`` over version *ranges*, so two builds a month
apart could differ in twenty transitive packages while the committed
``uv.lock`` described neither.  CI installed a torch range that the
package itself forbids.  These tests parse the four declarations with
the standard library (``tomllib``) plus ``packaging``/``yaml`` (both
already in the dependency graph) and fail whenever they drift apart.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
DOCKERFILE = REPO_ROOT / "Dockerfile"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Packages whose upgrade is a scientific decision, never an incidental one.
NUMERICAL_PACKAGES = ("torch", "torchvision", "timm", "numpy", "scikit-learn", "transformers")


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text())


@pytest.fixture(scope="module")
def lock_versions() -> dict[str, str]:
    lock = tomllib.loads(UV_LOCK.read_text())
    return {canonicalize_name(p["name"]): p["version"] for p in lock["package"]}


@pytest.fixture(scope="module")
def lock_project_metadata() -> dict:
    lock = tomllib.loads(UV_LOCK.read_text())
    (project,) = [p for p in lock["package"] if p["name"] == "prism-vrec"]
    return project


def _pip_specs(text: str) -> list[Requirement]:
    """Every quoted ``"name<spec>"`` argument of a ``pip install`` line."""
    specs: list[Requirement] = []
    for line in text.replace("\\\n", " ").splitlines():
        if "pip install" not in line:
            continue
        specs.extend(
            Requirement(token) for token in re.findall(r'"([A-Za-z0-9_.\-]+[<>=!~][^"]*)"', line)
        )
    return specs


def test_every_runtime_dependency_is_locked_inside_its_declared_range(pyproject, lock_versions):
    """The lock must be a valid resolution of the runtime dependency ranges."""
    requirements = [Requirement(dep) for dep in pyproject["project"]["dependencies"]]

    missing = [r.name for r in requirements if canonicalize_name(r.name) not in lock_versions]
    out_of_range = [
        f"{r.name}: locked {lock_versions[canonicalize_name(r.name)]} not in {r.specifier}"
        for r in requirements
        if canonicalize_name(r.name) in lock_versions
        and not r.specifier.contains(lock_versions[canonicalize_name(r.name)], prereleases=True)
    ]

    assert not missing, f"runtime dependencies absent from uv.lock: {missing}"
    assert not out_of_range, out_of_range


def test_numerical_packages_are_pinned_exactly_once_in_the_lock(lock_versions):
    """A single locked version per numerical library, parseable as PEP 440."""
    for name in NUMERICAL_PACKAGES:
        assert name in lock_versions, f"{name} is not in uv.lock"
        Version(lock_versions[name])


@pytest.mark.xfail(
    strict=True,
    reason=(
        "V01 finding: uv.lock predates the 'telemetry' extra (added in 2.6.0) and its "
        "psutil/nvidia-ml-py pins (7.2.2 / 13.610.43, resolved for codecarbon) fall outside "
        "the extra's '<7.0' / '<13.0' ranges. Regenerating the lock is a reviewed dependency "
        "operation; drop this marker when it lands."
    ),
)
def test_lock_records_the_same_requires_dist_as_pyproject(pyproject, lock_project_metadata):
    """``uv lock --check`` parity: extras and specifiers match the lock's metadata."""
    project = pyproject["project"]
    declared = {
        (canonicalize_name(Requirement(d).name), None, str(Requirement(d).specifier))
        for d in project["dependencies"]
    }
    for extra, deps in project.get("optional-dependencies", {}).items():
        declared |= {
            (canonicalize_name(Requirement(d).name), extra, str(Requirement(d).specifier))
            for d in deps
        }

    locked = set()
    for entry in lock_project_metadata["metadata"]["requires-dist"]:
        extra_match = re.search(r"extra == '([^']+)'", entry.get("marker", ""))
        locked.add(
            (
                canonicalize_name(entry["name"]),
                extra_match.group(1) if extra_match else None,
                entry.get("specifier", ""),
            )
        )

    assert declared == locked, {
        "only_in_pyproject": sorted(declared - locked),
        "only_in_lock": sorted(locked - declared),
    }


def test_dockerfile_installs_from_the_lock_not_from_ranges():
    """The image is built from ``uv.lock``; ranges are resolved nowhere at build time."""
    text = DOCKERFILE.read_text()

    assert re.search(r"^COPY .*\buv\.lock\b", text, re.MULTILINE), "uv.lock is not copied"
    assert "uv export --frozen" in text, "Dockerfile does not export the lock"
    assert "--require-hashes" in text, "locked install does not verify wheel hashes"
    assert not re.search(r'pip install (-e )?"?\.\[', text), "extras installed from ranges"


def test_dockerfile_base_python_satisfies_requires_python(pyproject):
    match = re.search(r"^FROM python:(\d+\.\d+)", DOCKERFILE.read_text(), re.MULTILINE)
    assert match, "Dockerfile base image is not an official python image"

    requires = Requirement(f"python{pyproject['project']['requires-python']}")

    assert requires.specifier.contains(match.group(1)), match.group(1)


def test_ci_installs_torch_inside_the_package_range_at_the_locked_version(pyproject, lock_versions):
    """CI's CPU-wheel pins must be the locked versions, inside pyproject's ranges."""
    declared = {
        canonicalize_name(Requirement(d).name): Requirement(d)
        for d in pyproject["project"]["dependencies"]
    }
    ci_specs = [s for s in _pip_specs(CI_WORKFLOW.read_text()) if s.name in NUMERICAL_PACKAGES]

    assert ci_specs, "ci.yml does not pin any numerical package explicitly"
    for spec in ci_specs:
        locked = lock_versions[canonicalize_name(spec.name)]
        assert spec.specifier == declared[spec.name].specifier or str(spec.specifier) == (
            f"=={locked}"
        ), f"{spec}: neither pyproject's {declared[spec.name].specifier} nor =={locked}"
        assert declared[spec.name].specifier.contains(locked), spec


def test_ci_ruff_pin_matches_the_lock(lock_versions):
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    steps = workflow["jobs"]["lint"]["steps"]
    (install,) = [s for s in steps if s.get("name") == "Install ruff"]
    (pin,) = re.findall(r"ruff==([0-9.]+)", install["run"])

    assert pin == lock_versions["ruff"]
