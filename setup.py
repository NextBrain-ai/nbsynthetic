
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

]

setup_requires = [
    'pytest-runner>=2.11.1',
]

test_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
]

development_requirements = [
    # general
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'm2r>=0.2.0',
    'Sphinx>=1.7.1',
    'sphinx_rtd_theme>=0.2.4',

    # style check
    'flake8>=3.5.0',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.1',
    'autopep8>=1.3.5',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1',
]

extras_require = {
    'test': test_require,
    'dev': development_requirements + test_require,
}

setup(
    name='nbsynthetic',
    keywords='nbsynthetic',
    version='0.1.0',   
    author="Javier Marin (NextBrain.ml)",
    author_email='javier.marin@softpoint.es',
    packages=find_packages(include=['nbsynthetic', 'nbsynthetic.*']),
    url='https://github.com/NextBrain-ml/nbsynthetic',
    license="MIT license",
    description="unsupervised synthetic data generator",
    classifiers=[
        'Development Status :: Beta',
        'Intended Audience :: Mide-level users',
        'License :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ], 
    extras_require=extras_require,
    entry_points={
        'console_scripts': [
            'nbsynthetic=nbsynthetic.cli:main'
        ]
    },
    install_package_data=True,
    install_requires=install_requires,
    license="MIT license",
    long_description=readme + '\n\n',
    long_description_content_type='text/markdown',
    include_package_data=True,
    python_requires='>=3.7',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=test_require,
    zip_safe=False,
)
