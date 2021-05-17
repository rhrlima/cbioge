from setuptools import find_packages, setup

install_requires = ['Keras==2.4.3', 'tensorflow==2.3.0', 
	'networkx==2.5', 'scipy==1.4.1', 'spektral==0.6.1', 
	'pandas==1.1.2', 'numpy==1.18.5', 'scikit-image']

current_vertion = '0.1.0'
setup(
	name='cbioge',
	packages=find_packages(),
	version=current_vertion,
	description='Cbio lib for grammar evolution',
	author='CBio Group',
	license='MIT',
	install_requires=install_requires,
	setup_requires=['pytest-runner'],
	tests_require=['pytest==4.4.1'],
	test_suite='tests',
	)
