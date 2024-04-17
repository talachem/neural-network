import setuptools

with open("README.md", "r") as fh:
    description = fh.read()

setuptools.setup(
    name="machineLearning",
    version="0.0.1",
    author="Johannes Bilk",
    author_email="johannes.bilk@physik.uni-giessen.de",
    description="A wholistic machine learning package used for pxd analysis",
    long_description=description,
    long_description_content_type="text/markdown",
    url="https://gitlab.ub.uni-giessen.de/gc2052/machine-learning",
    packages=setuptools.find_packages(),
    license='MIT',
    python_requires='>=3.11',
    install_requires=[
        "numpy>=1.21.0"
    ],
    keywords=['python', 'pxd', 'machine learning', 'decision tree', 'neural network', 'self-organizing map'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
