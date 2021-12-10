SHELL := /bin/bash
all: init activate

init:
	. ./init.sh
	virtualenv -p /usr/bin/python3 ./venv/
	chmod +x venv/bin/activate
activate:
	source ./venv/bin/activate; \
	cat requirements.txt | xargs -n 1 -L 1 pip3 install; \
	pip3 install -Ur requirements_parameter_search.txt; \

clean:
	rm -rf venv
