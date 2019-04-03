from setuptools import setup, find_packages

setup(name='trainer',
      version='0.1',
      packages=find_packages(),
      description='My First Run on Google Cloud',
      author='Jalil Ahmed',
      author_email='jalilmaqsood@gmail.com',
      license='MIT',
      install_requires=[
          'keras',
          'h5py'
      ],
      zip_safe=False)