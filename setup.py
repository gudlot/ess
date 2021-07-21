from setuptools import setup

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(name='ess',
      version='1.0',
      description='Neutron scattering tools for ESS',
      license="BSD-3-clause",
      long_description=long_description,
      author='Scipp contributors (https://github.com/scipp)',
      url="https://github.com/scipp/ess",
      packages=setuptools.find_packages("src"),
      package_dir={"": "src"},
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ])