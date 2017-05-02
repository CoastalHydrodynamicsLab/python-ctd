import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.verbose = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)


def extract_version(module='ctd'):
    version = None
    fdir = os.path.dirname(__file__)
    fnme = os.path.join(fdir, module, '__init__.py')
    with open(fnme) as fd:
        for line in fd:
            if (line.startswith('__version__')):
                _, version = line.split('=')
                # Remove quotation characters.
                version = version.strip()[1:-1]
                break
    return version

rootpath = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    return open(os.path.join(rootpath, *parts), 'r').read()


long_description = '{}\n{}'.format(read('README.rst'), read('CHANGES.txt'))
LICENSE = read('LICENSE.txt')


with open('requirements.txt') as f:
    require = f.readlines()
install_requires = [r.strip() for r in require]

setup(name='ctd',
      version=extract_version(),
      license=LICENSE,
      long_description=long_description,
      classifiers=['Development Status :: 5 - Production/Stable',
                  'Environment :: Console',
                  'Intended Audience :: Science/Research',
                  'Intended Audience :: Developers',
                  'Intended Audience :: Education',
                  'License :: OSI Approved :: BSD',
                  'Operating System :: OS Independent',
                  'Programming Language :: Python',
                  'Topic :: Scientific/Engineering',
                  'Topic :: Education',
                  ],
      description='Tools to load hydrographic data as DataFrames',
      author='Filipe Fernandes',
      author_email='ocefpaf@gmail.com',
      maintainer='Filipe Fernandes',
      maintainer_email='ocefpaf@gmail.com',
      url='https://github.com/pyoceans/python-ctd',
      download_url='http://pypi.python.org/pypi/ctd',
      platforms='any',
      keywords=['oceanography', 'data analysis', 'DataFrame'],
      install_requires=install_requires,
      tests_require=['pytest'],
      cmdclass=dict(test=PyTest),
      packages=['ctd', 'ctd/tests'],
      )
