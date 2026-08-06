#!/usr/bin/env python3
"""
Tests for the slackwater-rust workspace.

This is a Rust workspace with no Python code, but we can validate:
- Workspace Cargo.toml structure and member crates
- Each crate has valid Cargo.toml and source files
- Rust source files pass basic structural checks (no_empty_files, mod declarations)
- Cross-references between crates are consistent
- .coverage file is a valid SQLite database
- README and CHANGELOG exist and have expected sections
"""

import re
import sqlite3
from pathlib import Path

import pytest
import tomllib  # Python 3.11+

REPO_ROOT = Path(__file__).resolve().parent.parent


# ─── Workspace Cargo.toml Tests ───────────────────────────────────────────────

class TestWorkspaceCargo:
    """Validate the workspace-level Cargo.toml."""

    @staticmethod
    def _cargo():
        with open(REPO_ROOT / "Cargo.toml", "rb") as f:
            return tomllib.load(f)

    def test_cargo_toml_exists(self):
        assert (REPO_ROOT / "Cargo.toml").is_file()

    def test_has_workspace_section(self):
        data = self._cargo()
        assert "workspace" in data

    def test_has_resolver(self):
        data = self._cargo()
        assert data["workspace"].get("resolver") == "2"

    def test_has_members(self):
        data = self._cargo()
        members = data["workspace"].get("members", [])
        assert len(members) >= 5
        expected_prefix = "crates/"
        for m in members:
            assert m.startswith(expected_prefix), f"Member {m} doesn't start with {expected_prefix}"

    def test_expected_crates_listed(self):
        data = self._cargo()
        members = data["workspace"]["members"]
        expected_crates = [
            "flux-core",
            "swmidi",
            "tempo-core",
            "lattice-core",
            "tminus-core",
            "harmony-core",
            "perception-core",
        ]
        for crate in expected_crates:
            assert f"crates/{crate}" in members, f"Expected crate {crate} not in workspace members"

    def test_workspace_package_metadata(self):
        data = self._cargo()
        pkg = data["workspace"].get("package", {})
        assert "version" in pkg
        assert "edition" in pkg
        assert "license" in pkg
        assert pkg["license"] == "MIT"

    def test_workspace_dependencies(self):
        data = self._cargo()
        deps = data["workspace"].get("dependencies", {})
        assert "serde" in deps
        assert "serde_json" in deps
        assert "pyo3" in deps
        assert "rayon" in deps

    def test_workspace_authors(self):
        data = self._cargo()
        pkg = data["workspace"].get("package", {})
        assert "authors" in pkg
        assert "SuperInstance" in pkg["authors"]


# ─── Crate Structure Tests ────────────────────────────────────────────────────

class TestCrateStructure:
    """Validate each crate has proper structure."""

    EXPECTED_CRATES = [
        "flux-core",
        "swmidi",
        "tempo-core",
        "lattice-core",
        "tminus-core",
        "harmony-core",
        "perception-core",
    ]

    def test_all_crate_dirs_exist(self):
        for crate in self.EXPECTED_CRATES:
            crate_dir = REPO_ROOT / "crates" / crate
            assert crate_dir.is_dir(), f"Crate directory missing: {crate}"

    def test_all_crates_have_cargo_toml(self):
        for crate in self.EXPECTED_CRATES:
            cargo_path = REPO_ROOT / "crates" / crate / "Cargo.toml"
            assert cargo_path.is_file(), f"Crate {crate} missing Cargo.toml"

    def test_all_crates_have_src_dir(self):
        for crate in self.EXPECTED_CRATES:
            src_dir = REPO_ROOT / "crates" / crate / "src"
            assert src_dir.is_dir(), f"Crate {crate} missing src/ directory"

    def test_all_crates_have_lib_rs(self):
        for crate in self.EXPECTED_CRATES:
            lib_rs = REPO_ROOT / "crates" / crate / "src" / "lib.rs"
            assert lib_rs.is_file(), f"Crate {crate} missing src/lib.rs"

    def test_lib_rs_nonempty(self):
        for crate in self.EXPECTED_CRATES:
            lib_rs = REPO_ROOT / "crates" / crate / "src" / "lib.rs"
            content = lib_rs.read_text().strip()
            assert len(content) > 0, f"Crate {crate} has empty lib.rs"

    def test_crate_names_match_directory(self):
        for crate in self.EXPECTED_CRATES:
            cargo_path = REPO_ROOT / "crates" / crate / "Cargo.toml"
            with open(cargo_path, "rb") as f:
                data = tomllib.load(f)
            pkg_name = data.get("package", {}).get("name", "")
            # Cargo replaces - with _ in package names sometimes, but the name
            # in Cargo.toml should use hyphens for display
            assert crate in pkg_name or crate.replace("-", "_") in pkg_name, (
                f"Crate {crate} Cargo.toml has mismatched name: {pkg_name}"
            )


# ─── Implemented vs Placeholder Crates ────────────────────────────────────────

class TestCrateImplementation:
    """Distinguish implemented crates from placeholder stubs."""

    PLACEHOLDER_CRATES = ["swmidi", "tempo-core", "tminus-core", "perception-core"]
    IMPLEMENTED_CRATES = ["flux-core", "lattice-core", "harmony-core"]

    def test_placeholder_crates_have_marker(self):
        for crate in self.PLACEHOLDER_CRATES:
            lib_rs = REPO_ROOT / "crates" / crate / "src" / "lib.rs"
            content = lib_rs.read_text()
            assert "Placeholder" in content or "placeholder" in content, (
                f"Expected placeholder marker in {crate}/src/lib.rs"
            )

    def test_implemented_crates_have_substantive_code(self):
        for crate in self.IMPLEMENTED_CRATES:
            lib_rs = REPO_ROOT / "crates" / crate / "src" / "lib.rs"
            content = lib_rs.read_text()
            lines = [l for l in content.split("\n") if l.strip() and not l.strip().startswith("//")]
            assert len(lines) > 5, (
                f"Implemented crate {crate} lib.rs has too few non-comment lines: {len(lines)}"
            )

    def test_implemented_crates_have_module_declarations(self):
        """Implemented crates should declare modules."""
        for crate in self.IMPLEMENTED_CRATES:
            lib_rs = REPO_ROOT / "crates" / crate / "src" / "lib.rs"
            content = lib_rs.read_text()
            assert "pub mod" in content or "mod " in content, (
                f"Implemented crate {crate} has no module declarations"
            )


# ─── flux-core Source Tests ───────────────────────────────────────────────────

class TestFluxCoreSource:
    """Validate flux-core source files."""

    @staticmethod
    def _crate_dir():
        return REPO_ROOT / "crates" / "flux-core"

    def test_has_modules(self):
        src = self._crate_dir() / "src"
        modules = ["error_mask.rs", "exact.rs", "swmidi.rs"]
        for mod_name in modules:
            assert (src / mod_name).is_file(), f"flux-core missing module: {mod_name}"

    def test_lib_rs_exports(self):
        lib = (self._crate_dir() / "src" / "lib.rs").read_text()
        assert "pub mod error_mask" in lib
        assert "pub mod exact" in lib
        assert "pub mod swmidi" in lib

    def test_lib_rs_re_exports(self):
        lib = (self._crate_dir() / "src" / "lib.rs").read_text()
        assert "pub use" in lib

    def test_error_mask_has_eight_bits(self):
        """The error mask should define 8 friction dimensions."""
        error_mask = (self._crate_dir() / "src" / "error_mask.rs").read_text()
        # Count const definitions that look like flags
        flag_count = len(re.findall(r'const\s+\w+\s*[:=]', error_mask))
        assert flag_count >= 8, f"Expected at least 8 flag constants, got {flag_count}"

    def test_error_mask_has_serde(self):
        error_mask = (self._crate_dir() / "src" / "error_mask.rs").read_text()
        assert "Serialize" in error_mask or "Deserialize" in error_mask

    def test_exact_has_eisenstein_coord(self):
        exact = (self._crate_dir() / "src" / "exact.rs").read_text()
        assert "EisensteinCoord" in exact
        assert "pub struct" in exact

    def test_exact_has_type_aliases(self):
        exact = (self._crate_dir() / "src" / "exact.rs").read_text()
        for alias in ["Channel", "Pitch", "Velocity", "Tick", "Confidence"]:
            assert alias in exact, f"Missing type alias: {alias}"

    def test_swmidi_has_packing(self):
        swmidi = (self._crate_dir() / "src" / "swmidi.rs").read_text()
        assert "pack" in swmidi.lower() or "SwmidiEvent" in swmidi

    def test_has_integration_tests(self):
        test_file = self._crate_dir() / "tests" / "flux_test.rs"
        assert test_file.is_file()

    def test_has_benchmarks(self):
        bench_file = self._crate_dir() / "benches" / "packing.rs"
        assert bench_file.is_file()

    def test_deny_unsafe(self):
        lib = (self._crate_dir() / "src" / "lib.rs").read_text()
        assert "#![deny(unsafe_code)]" in lib

    def test_has_readme(self):
        assert (self._crate_dir() / "README.md").is_file()


# ─── harmony-core Source Tests ────────────────────────────────────────────────

class TestHarmonyCoreSource:
    """Validate harmony-core source files."""

    @staticmethod
    def _crate_dir():
        return REPO_ROOT / "crates" / "harmony-core"

    def test_has_modules(self):
        src = self._crate_dir() / "src"
        modules = ["phi.rs", "hurst.rs", "entropy.rs", "flow_state.rs", "cadence.rs", "protector.rs"]
        for mod_name in modules:
            assert (src / mod_name).is_file(), f"harmony-core missing module: {mod_name}"

    def test_lib_rs_exports_modules(self):
        lib = (self._crate_dir() / "src" / "lib.rs").read_text()
        for mod in ["cadence", "entropy", "hurst", "phi", "flow_state", "protector"]:
            assert f"pub mod {mod}" in lib, f"Missing pub mod {mod}"

    def test_has_hurst_exponent(self):
        hurst = (self._crate_dir() / "src" / "hurst.rs").read_text()
        assert "pub fn hurst_exponent" in hurst

    def test_has_phi_weights(self):
        phi = (self._crate_dir() / "src" / "phi.rs").read_text()
        assert "PhiWeights" in phi
        assert "pub fn compute_phi" in phi

    def test_has_flow_state_detector(self):
        flow = (self._crate_dir() / "src" / "flow_state.rs").read_text()
        assert "FlowStateDetector" in flow
        assert "FlowState" in flow

    def test_has_protection_action(self):
        protector = (self._crate_dir() / "src" / "protector.rs").read_text()
        assert "ProtectionAction" in protector
        assert "SuppressNotifications" in protector

    def test_has_entropy_function(self):
        entropy = (self._crate_dir() / "src" / "entropy.rs").read_text()
        assert "pub fn" in entropy

    def test_has_cadence_regularity(self):
        cadence = (self._crate_dir() / "src" / "cadence.rs").read_text()
        assert "pub fn cadence_regularity" in cadence

    def test_has_integration_tests(self):
        test_file = self._crate_dir() / "tests" / "harmony_test.rs"
        assert test_file.is_file()

    def test_has_benchmarks(self):
        bench_file = self._crate_dir() / "benches" / "hurst_bench.rs"
        assert bench_file.is_file()

    def test_deny_unsafe(self):
        lib = (self._crate_dir() / "src" / "lib.rs").read_text()
        assert "#![deny(unsafe_code)]" in lib

    def test_has_readme(self):
        assert (self._crate_dir() / "README.md").is_file()


# ─── lattice-core Source Tests ────────────────────────────────────────────────

class TestLatticeCoreSource:
    """Validate lattice-core source files."""

    @staticmethod
    def _crate_dir():
        return REPO_ROOT / "crates" / "lattice-core"

    def test_has_modules(self):
        src = self._crate_dir() / "src"
        modules = ["eisenstein.rs", "neighbors.rs", "region.rs", "snap.rs"]
        for mod_name in modules:
            assert (src / mod_name).is_file(), f"lattice-core missing module: {mod_name}"

    def test_lib_rs_exports(self):
        lib = (self._crate_dir() / "src" / "lib.rs").read_text()
        for mod in ["eisenstein", "neighbors", "region", "snap"]:
            assert f"pub mod {mod}" in lib

    def test_has_eisenstein_point(self):
        eisenstein = (self._crate_dir() / "src" / "eisenstein.rs").read_text()
        assert "EisensteinPoint" in eisenstein
        assert "pub struct" in eisenstein

    def test_has_neighbor_functions(self):
        neighbors = (self._crate_dir() / "src" / "neighbors.rs").read_text()
        assert "nearest_unoccupied" in neighbors
        assert "collides" in neighbors

    def test_has_lattice_region(self):
        region = (self._crate_dir() / "src" / "region.rs").read_text()
        assert "LatticeRegion" in region

    def test_has_snap_functions(self):
        snap = (self._crate_dir() / "src" / "snap.rs").read_text()
        assert "snap_position" in snap
        assert "snap_all" in snap

    def test_has_integration_tests(self):
        test_file = self._crate_dir() / "tests" / "lattice_test.rs"
        assert test_file.is_file()

    def test_has_benchmarks(self):
        bench_file = self._crate_dir() / "benches" / "snap_bench.rs"
        assert bench_file.is_file()

    def test_deny_unsafe(self):
        lib = (self._crate_dir() / "src" / "lib.rs").read_text()
        assert "#![deny(unsafe_code)]" in lib

    def test_has_readme(self):
        assert (self._crate_dir() / "README.md").is_file()


# ─── Integration Test File Tests ──────────────────────────────────────────────

class TestIntegrationTestFiles:
    """Validate Rust integration test files."""

    TEST_FILES = [
        ("flux-core", "tests/flux_test.rs"),
        ("harmony-core", "tests/harmony_test.rs"),
        ("lattice-core", "tests/lattice_test.rs"),
    ]

    def test_integration_tests_exist(self):
        for crate, path in self.TEST_FILES:
            full_path = REPO_ROOT / "crates" / crate / path
            assert full_path.is_file(), f"Missing integration test: {crate}/{path}"

    def test_integration_tests_have_test_functions(self):
        for crate, path in self.TEST_FILES:
            full_path = REPO_ROOT / "crates" / crate / path
            content = full_path.read_text()
            test_count = len(re.findall(r'#\[test\]', content))
            assert test_count >= 3, (
                f"{crate} integration tests only have {test_count} #[test] functions"
            )

    def test_integration_tests_have_assertions(self):
        for crate, path in self.TEST_FILES:
            full_path = REPO_ROOT / "crates" / crate / path
            content = full_path.read_text()
            assert "assert" in content, f"{crate} integration tests have no assertions"


# ─── .coverage File Tests ─────────────────────────────────────────────────────

class TestCoverageFile:
    """Validate the .coverage SQLite database."""

    def test_coverage_file_exists(self):
        assert (REPO_ROOT / ".coverage").is_file()

    def test_coverage_is_valid_sqlite(self):
        db_path = REPO_ROOT / ".coverage"
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [t[0] for t in tables]
            assert "coverage_schema" in table_names
            assert "file" in table_names
            assert "context" in table_names
        finally:
            conn.close()

    def test_coverage_version(self):
        db_path = REPO_ROOT / ".coverage"
        conn = sqlite3.connect(str(db_path))
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key='version'")
            result = cursor.fetchone()
            assert result is not None
            version = result[0]
            # Should be a reasonable coverage.py version
            assert version.startswith(("4.", "5.", "6.", "7."))
        finally:
            conn.close()


# ─── Documentation Tests ──────────────────────────────────────────────────────

class TestDocumentation:
    """Validate workspace documentation."""

    def test_readme_exists(self):
        assert (REPO_ROOT / "README.md").is_file()

    def test_readme_mentions_slackwater(self):
        readme = (REPO_ROOT / "README.md").read_text()
        assert "slackwater" in readme.lower() or "Slackwater" in readme

    def test_changelog_exists(self):
        assert (REPO_ROOT / "CHANGELOG.md").is_file()

    def test_license_exists(self):
        assert (REPO_ROOT / "LICENSE").is_file()

    def test_gitignore_exists(self):
        assert (REPO_ROOT / ".gitignore").is_file()

    def test_gitignore_has_target(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text()
        assert "target" in gitignore or "/target" in gitignore

    def test_cargo_lock_exists(self):
        assert (REPO_ROOT / "Cargo.lock").is_file()


# ─── Safety/Quality Attribute Tests ───────────────────────────────────────────

class TestSafetyAttributes:
    """Check Rust safety and quality attributes across all source files."""

    @staticmethod
    def _all_rs_files():
        return list((REPO_ROOT / "crates").rglob("*.rs"))

    def test_all_rs_files_nonempty(self):
        for rs_file in self._all_rs_files():
            content = rs_file.read_text().strip()
            assert len(content) > 0, f"Empty Rust file: {rs_file}"

    def test_implemented_crates_deny_unsafe(self):
        """Implemented crate lib.rs files should deny unsafe code."""
        implemented = ["flux-core", "lattice-core", "harmony-core"]
        for crate in implemented:
            lib = REPO_ROOT / "crates" / crate / "src" / "lib.rs"
            content = lib.read_text()
            assert "#![deny(unsafe_code)]" in content, (
                f"{crate} should deny unsafe code"
            )

    def test_implemented_crates_warn_clippy(self):
        """Implemented crate lib.rs files should warn on clippy."""
        implemented = ["flux-core", "lattice-core", "harmony-core"]
        for crate in implemented:
            lib = REPO_ROOT / "crates" / crate / "src" / "lib.rs"
            content = lib.read_text()
            assert "#![warn(clippy::all)]" in content, (
                f"{crate} should warn on clippy::all"
            )

    def test_no_unsafe_blocks_in_source(self):
        """No unsafe blocks in any source file (outside tests)."""
        for rs_file in self._all_rs_files():
            if "/tests/" in str(rs_file):
                continue
            content = rs_file.read_text()
            unsafe_blocks = re.findall(r'unsafe\s*\{', content)
            # Allow the deny attribute but no actual unsafe blocks
            assert len(unsafe_blocks) == 0, (
                f"Unsafe block found in {rs_file}"
            )
