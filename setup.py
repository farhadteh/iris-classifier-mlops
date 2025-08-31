"""
Setup script for iris_classifier package
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="iris-classifier",
    version="1.0.0",
    author="MLOps Team",
    author_email="mlops@example.com",
    description="A comprehensive MLOps pipeline for Iris flower classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/iris-classifier",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "iris-train=iris_classifier.training.train:main",
            "iris-serve=iris_classifier.api.fastapi_app:main",
            "iris-demo=scripts.demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "iris_classifier": ["configs/*.yaml", "configs/*.json"],
    },
    zip_safe=False,
)
