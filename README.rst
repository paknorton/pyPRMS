pyPRMS
======

A Python library for working with the Precipitation-Runoff Modeling System (PRMS)

----

The pyPRMS library provides a set of python interfaces to read, modify, and write various files used by and for PRMS.

This library requires numpy >= 1.13.0 and pandas >= 0.20.0.

Anaconda install
----------------
A virtual environment for using pyPRMS can be setup by putting the following in a YAML file named pyprms_env.yml::

    name: prms
    dependencies:
    - python=2.7.*
    - anaconda
    - future
    - jupyter
    - numpy>=1.13
    - pandas>=0.20
    - pip
    - pip:
        - https://github.com/paknorton/pyPRMS

and executing::

    conda env create -f pyprms_env.yml

pip install
-----------
The library can be installed using pip. To install from the master branch::

    pip install https://github.com/paknorton/pyPRMS

To install from development branch use the following command::

    pip install https://github.com/paknorton/pyPRMS/zipball/development

To update from the development branch type::

    pip install https://github.com/paknorton/pyPRMS/zipball/development --upgrade

To uninstall pyPRMS type::
    pip uninstall pyPRMS


