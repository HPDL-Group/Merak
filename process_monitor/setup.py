from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="procguard",
    version="0.1.0",
    author="ProcGuard Team",
    author_email="procguard@example.com",
    description="A high-availability process monitoring and recovery system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/procguard",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "psutil>=5.9.0",
        "PyYAML>=6.0",
        "pyzmq>=25.0.0",
    ],
    entry_points={
        "console_scripts": [
            "procguard=procguard.procguard:serve",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
