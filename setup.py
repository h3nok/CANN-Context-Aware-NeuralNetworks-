from setuptools import setup

setup(
    name='deepclo',
    version='1.0',
    packages=['deepclo',
              'deepclo.core',
              'deepclo.core.measures',
              'deepclo.pipe',
              'deepclo.models',
              'deepclo.algorithms'],
    url='https://github.com/h3nok/curriculum_learning_optimization.git',
    license='MIT',
    author='Henok',
    author_email='henok.ghebrechristos@ucdenver.edu',
    description='Deep Curriculum Learning Optimization'
)
