import setuptools

setuptools.setup(name='CCLya-Payne',
      version='1.1',
      description='Tools for emulating Lyman-alpha profiles with neural networks.',
      author='Erik Solhaug, Hsiao-Wen Chen',
      author_email='eriksolhaug@uchicago.edu',
      license='MIT',
      url='https://github.com/highzclouds/CCLya-Payne',
      packages=['cclya_payne', 'cclya_payne/fitting', 'cclya_payne/training', 'cclya_payne/utils'],
      package_data={'cclya_payne':['data/*']},
      install_requires=['torch', 'torchvision', 'numpy', 'pandas', 'matplotlib', 'emcee', 'corner', 'scipy','radam @ git+https://github.com/LiyuanLucasLiu/RAdam.git'])
