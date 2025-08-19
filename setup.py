from setuptools import setup, find_packages

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='eyemovement_data',
    url='https://github.com/lukekorthals/pursuing-smooth-pursuits-data',
    author='Luke Korthals',
    author_email='luke-korthals@outlook.de',
    packages=find_packages(),
    package_data={"eyemovement_data": ["clean_data.R"]},
    include_package_data=True,
    install_requires=[
        'ipykernel',
        'numpy',
        'matplotlib',
        'osfclient',
        'pandas',
        'scikit-learn',
        ],
    version='0.1',
    license='MIT',
    description='Eye tracking data collected 2023 at the University of Amsterdam.',
    long_description='WIP',
)