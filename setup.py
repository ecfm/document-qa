from setuptools import setup, find_packages

setup(
    name="document-qa",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "streamlit",
        "pandas",
        "datasets"
    ],
    python_requires=">=3.7",
) 