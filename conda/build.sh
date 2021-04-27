set -ex
echo __version__ = '"'$(python src/ess/_version.py)'"' > _fixed_version.py
cp _fixed_version.py src/ess/ 
cp -r src/ess "$CONDA_PREFIX"/lib/python*/
