"""
Shared test fixtures and configuration for MRtrix3 Agent tests.

This module provides common fixtures used across unit, integration, and e2e tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch


def pytest_configure(config):
    """
    Called before test collection starts.
    """
    # Always set defaults for values that don't have secrets (needed for imports)
    os.environ.setdefault("EMBEDDING_MODEL", "models/gemini-embedding-001")
    os.environ.setdefault("EMBEDDING_DIMENSIONS", "768")

    # For URL and KEY, use setdefault to allow real secrets in CI
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test_key")
    os.environ.setdefault("GOOGLE_API_KEY", "test_gemini_key")
    os.environ.setdefault("GOOGLE_API_KEY_EMBEDDING", "test_embedding_key")


@pytest.fixture
def mock_env(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "test_key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test_gemini_key")
    monkeypatch.setenv("GOOGLE_API_KEY_EMBEDDING", "test_embedding_key")
    monkeypatch.setenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_rst_content():
    """Sample RST content for testing document processing."""
    return """
dwiextract
==========

Synopsis
--------

Extract individual volumes (or a subset of volumes) from DWI data

Usage
-----

``dwiextract input output [options]``

Description
-----------

This command is used to extract individual volumes from DWI data.

Options
-------

``-fslgrad bvecs bvals``
  Provide the diffusion-weighted gradient scheme used in the acquisition

``-grad file``
  Provide the diffusion-weighted gradient scheme used in the acquisition

``-bzero``
  Extract only the b=0 volumes

Examples
--------

Extract volumes corresponding to a particular b-value::

    dwiextract dwi.mif dwi_b0.mif -bzero

References
----------

Tournier, J.-D.; Smith, R. E.; Raffelt, D.; Tabbara, R.; Dhollander, T.; Pietsch, M.; Christiaens, D.; Jeurissen, B.; Yeh, C.-H. & Connelly, A. MRtrix3: A fast, flexible and open software framework for medical image analysis and visualisation. NeuroImage, 2019, 202, 116137

Author: Robert E. Smith (robert.smith@florey.edu.au)

Copyright: Copyright (c) 2008-2024 the MRtrix3 contributors.
"""


@pytest.fixture
def sample_commands_list_rst():
    """Sample commands_list.rst content for testing command parsing."""
    return """
Commands list
=============

.. toctree::
   :maxdepth: 1

   commands/5ttgen
   commands/amp2sh
   commands/connectome2tck
   commands/dirgen
   commands/dwiextract
   commands/dwi2fod
   commands/mrcalc
   commands/mrconvert
   commands/mrstats
   commands/tckgen
   commands/tcksample
   commands/transformcalc
"""


@pytest.fixture
def mock_git_subprocess():
    """Mock subprocess calls for git operations."""
    with patch("subprocess.run") as mock_run:
        # Configure successful git operations
        mock_run.return_value.returncode = 0
        yield mock_run


@pytest.fixture
def sample_command_files():
    """Sample command RST files that should exist in commands/ directory."""
    return [
        "5ttgen.rst",
        "amp2sh.rst",
        "connectome2tck.rst",
        "dirgen.rst",
        "dwiextract.rst",
        "dwi2fod.rst",
        "mrcalc.rst",
        "mrconvert.rst",
        "mrstats.rst",
        "tckgen.rst",
        "tcksample.rst",
        "transformcalc.rst",
    ]
