"""
Setup of core python codebase
Author: Jeff Mahler
"""
from setuptools import setup

requirements = [
    'numpy',
]

exec(open('registration/version.py').read())


setup(
    name='registration',
    version = __version__,
    description = 'Registration utilities for the Berkeley AutoLab',
    long_description = 'Registration utilities for the Berkeley AutoLab',
    author = 'Matthew Matl',
    author_email = 'mmatl@berkeley.edu',
    license = 'Apache Software License',
    url = 'https://github.com/BerkeleyAutomation/registration',
    keywords = 'robotics grasping transformations registration',
    classifiers = [
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering'
    ],
    packages = ['registration'],
    install_requires = requirements,
    extras_require = { 'docs' : [
                            'sphinx',
                            'sphinxcontrib-napoleon',
                            'sphinx_rtd_theme'
                        ],
                       'ros' : [
                           'rospkg',
                           'catkin_pkg',
                           'empy'
                        ],
    }
)

