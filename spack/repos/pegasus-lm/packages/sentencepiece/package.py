# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

# ----------------------------------------------------------------------------
# If you submit this package back to Spack as a pull request,
# please first remove this boilerplate and all FIXME comments.
#
# This is a template package file for Spack.  We've put "FIXME"
# next to all the things you'll want to change. Once you've handled
# them, you can save this file and test your package like this:
#
#     spack install sentencepiece
#
# You can edit this file again by typing:
#
#     spack edit sentencepiece
#
# See the Spack documentation for more information on packaging.
# ----------------------------------------------------------------------------

from spack.package import *


class Sentencepiece(CMakePackage):
    """Unsupervised text tokenizer for Neural Network-based text generation.

    This is the C++ package.
    """

    homepage = "https://github.com/google/sentencepiece"
    url = "https://github.com/google/sentencepiece/archive/v0.1.85.tar.gz"

    version("0.1.97", sha256="41c3a07f315e3ac87605460c8bb8d739955bc8e7f478caec4017ef9b7d78669b")
    version("0.1.91", sha256="acbc7ea12713cd2a8d64892f8d2033c7fd2bb4faecab39452496120ace9a4b1b")
    version("0.1.85", sha256="dd4956287a1b6af3cbdbbd499b7227a859a4e3f41c9882de5e6bdd929e219ae6")

    variant('gperftools', default=True, description='optional, high-performance malloc() impl')

    depends_on("cmake@3.1:", type="build")
    depends_on("gperftools", when="+gperftools")  # optional, 10-40% performance improvement
