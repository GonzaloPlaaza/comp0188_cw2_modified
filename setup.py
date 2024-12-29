from distutils.core import setup
import pathlib
import setuptools


HERE = pathlib.Path(__file__).parent

setuptools.setup(
    name='comp0188_cw2',
    version='1.0.0',
    description="",
    long_description="",
    packages=setuptools.find_packages(where="src"),
    author="",
    author_email="",
    long_description_content_type="text/markdown",
    url="",
    license='MIT',
    classifiers=[],
    package_dir={"": "src"},
    python_requires="",
    install_requires=[""]
)