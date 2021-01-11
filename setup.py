import setuptools


with open("README.md", "r") as fh:
	long_description = fh.read()
	

setuptools.setup(
	name='boxeye',
	version='0.1',
	url="https://github.com/bot-boi/box-eye",
	packages=['boxeye'],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
	],
)