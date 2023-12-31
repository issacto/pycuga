import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycuga",
    version="0.0.13",
    author="Issac To",
    description="Python Cuda Genetic Algorithm Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/issacto/PyCuGa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[         
        'pandas',         
        'numpy',
        'pycuda',
        'matplotlib',
    ],
)
