
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]

setup(name='projection-pursuit',
      version='0.4',
      description='Scikit-learn estimators based on projection pursuit.',
      url='https://github.com/pavelkomarov/projection-pursuit',
      author='Pavel Komarov',
      license='BSD',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='pvlkmrv@gmail.com')
