Changelog
=========

v0.1.5 (2023-08-21)
-------------------

### Refactoring
* transpose all output matrices to fit with Antares expected time-series shape

v0.1.4 (2023-08-09)
-------------------

### Bug fixes
* various code fixes inside module `ts_generator.py` to match with the existing C++ code

### Checkstyle
* add line-length limit of 120 for black and isort


v0.1.3 (2023-08-05)
-------------------

* Remove unused imports
* Use 8760 values for `cluster.modulation` instead of 24.

v0.1.2 (2023-08-02)
-------------------

* Add `py.typed` file to avoid mypy issue in projects importing the code
* Fix little typo inside `README.md` file.

v0.1.1 (2023-08-02)
-------------------

* Add project description to publish on PyPi.

v0.1.0 (2023-08-02)
-------------------

* First release of the project.