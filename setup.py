from setuptools import setup

setup(name='CCLya-Payne',
      version='1.1',
      description='Tools for emulating Lyman-alpha profiles with neural networks.',
      author='Erik Solhaug',
      author_email='eriksolhaug@uchicago.edu',
      license='MIT',
      url='https://github.com/highzclouds/CCLya-Payne',
      package_dir = {},
      packages=['CCLya-Payne'],
      package_data={'CCLya-Payne':['data/*.npz']},
      dependency_links = [],
      install_requires=['torch', 'torchvision'])
