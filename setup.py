import setuptools


with open("README.md", "r") as fh:
	long_description = fh.read()
	

setuptools.setup(
	name='box-eye',
	version='0.1',
	# scripts=['src'],
	author="bot-boi & bluecereal",
	author_email="N/A",
	description="A highly constrain-able computer vision bot framework for Android emulators.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/bot-boi/box-eye",
	packages={'box_eye': 'src'},
	package_dir={'box_eye': 'src'},
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)