# Conda export script with pip dependencies: https://github.com/conda/conda/issues/9628#issuecomment-1608913117
# Extract installed pip packages
pip_packages=$(conda env export | grep -A9999 ".*- pip:" | grep -v "^prefix: ")

# Export conda environment without builds, and append pip packages
conda env export --from-history | grep -v "^prefix: " > environment.yml
echo "$pip_packages" >> environment.yml