from setuptools import find_packages, setup

install_requires = [
    'numpy', 
    'scikit-image', 
    'tensorflow==1.14', 
    'Keras==2.2.5', 
    # 'scipy==1.4.1', 
    # 'spektral==0.6.1', 
	# 'pandas==1.1.2', 
    # 'networkx==2.5', 
]

current_vertion = '0.1.1'
setup(
	name='cbioge',
	packages=find_packages(include=['cbioge']),
	version=current_vertion,
	description='CBio lib for grammar evolution',
	author='CBio Group',
	license='MIT',
	install_requires=install_requires,
	setup_requires=['pytest-runner'],
	tests_require=['pytest>=4.4.1'],
	test_suite='cbioge/tests',
)