"""Setup script for SAS Viya Autoencoder Python package."""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Autoencoder implementation in Python using PyTorch"

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name="sas-viya-autoencoder",
    version="0.1.0",
    author="SAS Institute",
    author_email="support@sas.com",
    description="Python implementation of autoencoders for SAS Viya",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sassoftware/enlighten-apply",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Data Scientists",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0,<8.1.0",
            "pytest-cov>=4.1.0,<5.1.0",
            "black>=23.7.0,<24.5.0",
            "flake8>=6.0.0,<7.1.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0,<1.1.0",
            "ipython>=8.12.0,<8.25.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sas-autoencoder=src.scripts.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
