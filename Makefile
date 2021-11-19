SHELL := /bin/bash
all: init activate

init:
	. ./init.sh
	/usr/bin/python3 -m venv ./venv/
	chmod +x venv/bin/activate
activate:
	source ./venv/bin/activate; \
	cat requirements.txt | xargs -n 1 -L 1 pip3 install; \
	pip3 install -Ur requirements_parameter_search.txt; \

clean:
	rm -rf venv
