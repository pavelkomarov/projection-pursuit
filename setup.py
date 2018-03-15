
from setuptools import setup, find_packages

setup(name='projection-pursuit',
      version='0.6',
      description='Scikit-learn estimators based on projection pursuit.',
      url='https://github.com/pavelkomarov/projection-pursuit',
      author='Pavel Komarov',
      license='BSD',
      packages=[x for x in find_packages() if 'tests' not in x],
      install_requires=['pytest', 'scikit-learn', 'numpy', 'matplotlib'],
      author_email='pvlkmrv@gmail.com')
