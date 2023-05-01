from setuptools import setup, Command
import os

dependencies = [
    'pandas',
    'scikit-learn',
]


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('del -vrf ./*.pyc ./*.tgz ./*.egg-info ./test-reports ./.pytest_cache')


class UploadCommand(Command):
    """Custom clean command to tidy up the project root."""
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        pass


setup(
    name='deepclo',
    version='1.0',
    packages=['deepclo',
              'deepclo.core',
              'deepclo.core.measures',
              'deepclo.pipe',
              'deepclo.models',
              'deepclo.algorithms',
              'experiments',
              'experiments.adversary',
              'experiments.synthetic'],
    url='https://github.com/h3nok/curriculum_learning_optimization.git',
    license='MIT',
    author='Henok',
    author_email='henok.ghebrechristos@ucdenver.edu',
    description='Deep Curriculum Learning Optimization',

    cmdclass={
        'clean': CleanCommand,
        'upload': UploadCommand
    }
)
