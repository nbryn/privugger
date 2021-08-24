from setuptools import setup, find_packages
 

#pip install -e . to test locally before publishing
#build command: python3 setup.py bdist_wheel sdist
#push to PyPi command: twine upload dist/*


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name='privugger',
    version='0.0.1',
    author='Raúl Pardo, Mathias Valdbjørn Jørgensen, and Rasmus Carl Rønneberg',
    author_email='raup@itu.dk',
    description='Privacy risk analysis library for Python programs.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/itu-square/privugger',
    package_dir={"": "privugger"},
    packages=find_packages(where="privugger"),
)
