# nlp-techniques

This suite of models was tested on both the production machine as well as a fresh install of Ubuntu 20.0.4.2 LTS. All modules were capable of being run on a system with 8GB of RAM.

Installation of the necessary packages can be done by:

    navigating to this folder in the teriminal and running "pip3 install -r requirements.txt"
    upgrade tensorflow (pip3 install tensorflow==2.6.0)
    install graphviz

Given that there are 10 modules which output data, "group_project.py" has been configured as a switchboard of sorts to handle which modules are to be run by commenting or uncommenting as appropriate.

Each module listed in group_project will output a model summary and output statistics in the appropriate subfolder in the "metrics" folder. All but the Bert and sentiment analysis models (based on Roberta) will also output a graph of the model structure. The Bert and Roberta models were too large to graphically represent.
