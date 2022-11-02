
install: 
	python setup.py bdist_wheel
	python -m pip install dist/* --force-reinstall


all:
	install
