#!/usr/bin/env python3
"""
Setup script for Whisper Domain Adaptation package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="whisper-domain-adaptation",
    version="1.0.0",
    author="Whisper Domain Adaptation Team",
    author_email="your.email@example.com",
    description="Domain adaptation framework for Whisper speech recognition models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisper-domain-adaptation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/whisper-domain-adaptation/issues",
        "Source": "https://github.com/yourusername/whisper-domain-adaptation",
        "Documentation": "https://github.com/yourusername/whisper-domain-adaptation/tree/main/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "whisper-domain-train=whisper_dyslexia_cross_domain:main",
            "whisper-domain-test=test_whisper_model:main",
            "whisper-domain-eval=test_dyslexia_dataset:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "speech recognition",
        "whisper",
        "domain adaptation",
        "dyslexia",
        "machine learning",
        "deep learning",
        "audio processing",
        "korean language",
    ],
    license="MIT",
    zip_safe=False,
)
