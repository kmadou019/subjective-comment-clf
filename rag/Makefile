.PHONY : run clean install require 

run:
	cd script && ./script_rag.sh

clean:
	rm graph_builder_key.py graph_builder_no.py rag_key.py rag_no.py log/* excel/* script/OAR*

install:
	pip install -r require.txt

require:
	pip freeze > require.txt

