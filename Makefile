
install: 
	python setup.py bdist_wheel
	python -m pip install dist/* --force-reinstall

build: 
	(cd src && make pybind && mv pyticg* ../pylib)

all:
	build
	install
