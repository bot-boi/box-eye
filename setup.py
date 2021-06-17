import setuptools


setuptools.setup(
	name='boxeye',
	version='0.0.5',
	url="https://github.com/bot-boi/box-eye",
	packages=['boxeye', 'boxeye/botutils'],
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
	],
    install_requires=[
        'numpy',
        'opencv-python',
        'Pillow',
        'pure-python-adb',
	    'pytesseract',
        'pytest',
        'vectormath',
    ],
)
