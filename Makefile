
install: 
	python setup.py bdist_wheel
	python -m pip install dist/* --force-reinstall

build: 
	(cd src && make pybind && mv pyticg* ../pylib)

clean:
	rm -r build
	rm -r dist

all:
	build
	install
