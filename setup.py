from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()


install_requires = [
    'numpy>=1.21.6',
    'pandas>=1.3.5',
    'scikit-learn>=1.0.2',
    'scipy>=1.7.3',
    'tensorflow>=2.8.2',
    'keras>=2.8.0',
    'tqdm>=4.64.0',
    'plotly>=5.5.0',
    'ripser>=0.6.4'

]

setup_requires = [
    'pytest-runner>=6.0.0',
]

test_require = [
    'pytest>=7.1.2',
    'pytest-cov>=3.0.0',
]

development_requirements = [
    # general
    'bumpversion>=0.6.0',
    'pip>=22.2.2',
    'watchdog>=2.1.9',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.5.1',
    'autopep8>=1.7.0',

    # distribute on PyPI
    'twine>=4.0.1',
    'wheel>=0.37.1',

]

extras_require = {
    'test': test_require,
    'dev': development_requirements + test_require,
}

setup(
    name='nbsynthetic',
    keywords='nbsynthetic',
    version='0.1.1',   
    author="Javier Marin (NextBrain.ml)",
    author_email='javier.marin@softpoint.es',
    packages=find_packages(include=['nbsynthetic', 'nbsynthetic.*']),
    url='https://github.com/NextBrain-ml/nbsynthetic',
    license="MIT license",
    description="Unsupervised synthetic data generator",
    classifiers=[
        'Development Status :: Pre Alpha',
        'Intended Audience :: Mide-level users',
        'License :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ], 
    extras_require=extras_require,
    #entry_points={
    #    'console_scripts': [
    #        'nbsynthetic=nbsynthetic.cli:main'
    #    ]
    #},
    install_requires=install_requires,
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    include_package_data=True,
    python_requires='>=3.7',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=test_require,
    zip_safe=False,
)
