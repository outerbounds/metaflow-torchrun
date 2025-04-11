from setuptools import setup, find_namespace_packages

version = "0.1.2"

setup(
    name="metaflow-torchrun",
    version=version,
    description="A torchrun decorator for Metaflow",
    author="Outerbounds",
    author_email="hello@outerbounds.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include=["metaflow_extensions.*"]),
    py_modules=[
        "metaflow_extensions",
    ],
    install_requires=[],
)
