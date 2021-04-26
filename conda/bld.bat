@echo off
echo | set /p="__version__ = '" > _fixed_version.py
python src\ess\_version.py > output_ver
set /p version=<output_ver
echo | set /p=%version%' >> _fixed_version.py
del output_ver
move _fixed_version.py src\ess
move src\ess %CONDA_PREFIX%\lib\
