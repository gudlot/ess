@echo off
echo | set /p="__version__" > _fixed_version.py
echo | set /p="=" >> _fixed_version.py
echo | set /p=$(python src/ess/_version.py) >> _fixed_version.py
move _fixed_version.py src\ess
move src\ess %CONDA_PREFIX%\lib\
