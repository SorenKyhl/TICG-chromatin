
install:
	python setup.py bdist_wheel
	python -m pip install dist/* --force-reinstall

build:
	(cd src && make pybind && mv pyticg* ../pylib)

clean:
	rm -r build
	rm -r dist

all:
	make clean && make build && make install

docs:
	(cd pylib/docs && python -m sphinx.cmd.build -M html source/ build/)
