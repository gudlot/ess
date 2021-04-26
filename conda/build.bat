@echo off

echo __version__ = '"'%undefined%'"' REM UNKNOWN: {"type":"Redirect","op":{"text":">","type":"great"},"file":{"text":"_fixed_version.py","type":"Word"}}
COPY  _fixed_version.py src\ess\
COPY  src\ess "%CONDA_PREFIX%"\lib\
