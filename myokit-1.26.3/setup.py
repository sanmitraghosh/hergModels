from setuptools import setup, find_packages
from myokit import VERSION as version


# Load text for description and license
with open('README') as f:
    readme = f.read()
with open('LICENSE') as f:
    license = f.read()


# Go!
setup(
    # Module name
    name='myokit',

    # Version
    version=version,

    description='A simple interface to cardiac cellular electrophysiology',

    long_description=readme,

    license=license,

    author='Michael Clerx',

    author_email='michael@myokit.org',

    url='http://myokit.org',

    # Packages to include
    packages=find_packages(include=('myokit', 'myokit.*')),

    # List of dependencies
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib>=1.5',
        # PyQT or PySide?
    ],
)
