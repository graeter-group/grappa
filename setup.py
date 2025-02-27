from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    data_files=[('data', ['src/grappa/data/published_datasets.csv']),
                ('models', ['src/grappa/models/published_models.csv'])],
)

