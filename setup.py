from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()


DESCRIPTION = 'Determine number of principle components based on sequencing data'
LONG_DESCRIPTION = 'A package that determines the number of top informative principal components based on sequencing data.'

# Setting up
setup(
    name="ERStruct",
    version='0.1.3',
    license='MIT',
    author="Jinghan Yang",
    author_email="<eciel@connect.hku.hk>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    url='https://github.com/ecielyang/ERStruct',
    keywords=['Population structure', 'Principal component', 'Random matrix theory', 'Sequencing data', 'Spectral analysis'],
    packages=find_packages(),
    install_requires=['numpy', 'torch', 'joblib'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ]
)