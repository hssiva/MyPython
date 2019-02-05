from setuptools import setup, find_packages
import sys, os

version = '0.0.1'

setup(name='MyPython',
      version=version,
      description="a simple python machine learning example",
      long_description="""\
""",
      classifiers=[], # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
      keywords='machine-learing,  decision-trees random-forest gradient-boosting support-vector-machines',
      author='Siva',
      author_email='halasya.siva@gmail.com',
      url='',
      license='MIT',
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=[
          # -*- Extra requirements: -*-
      ],
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
