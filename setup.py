from setuptools import setup, find_packages

# Read requirements.txt and use it for the install_requires parameter
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='autoencoder_optimiser',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    # Additional metadata about your package
    author='Aygalic Jara--Mikolajczak',
    author_email='aygalic@hotmail.fr',
    description='A Genetic Algorithm encapsulated approach to auto encoders',
    # More metadata
)
