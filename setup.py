from setuptools import setup

setup(name='CCLya-Payne',
      version='1.1',
      description='Tools for emulating Lyman-alpha profiles with neural networks.',
      author='Erik Solhaug, Hsiao-Wen Chen',
      author_email='eriksolhaug@uchicago.edu',
      license='MIT',
      url='https://github.com/highzclouds/CCLya-Payne',
      package_dir = {},
      packages=['cclya_payne', 'cclya_payne/fitting', 'cclya_payne/training', 'cclya_payne/utils'],
      package_data={'cclya_payne':['data/*']},
      dependency_links = [],
      install_requires=['torch', 'torchvision', 'math', 'pickle', 'os', 'numpy', 'time', `radam @ git+https://github.com/LiyuanLucasLiu/RAdam.git', 'pandas', 'matplotlib', 'emcee', 'corner', scipy'])
