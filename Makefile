SHELL := /bin/bash
all: init activate

init:
	virtualenv -p /usr/bin/python3 venv
	chmod +x venv/bin/activate
activate:
	source ./venv/bin/activate; \
	cat requirements.txt | xargs -n 1 -L 1 pip3 install; \

clean:
	rm -rf venv
