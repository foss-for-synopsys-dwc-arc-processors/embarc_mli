embARC MLI Documentation
==================================================

The embARC MLI documentation can be built using [Sphynx](http://sphinx-doc.org/) together with theme provided by [Read the Docs](https://readthedocs.org/)

To build the documentation you first need to install python. See [this instruction](/examples/tutorial_emnist_tensorflow#install-python-and-create-a-virtual-environment) as one of the ways to do so.

Requirements for building the embARC documentation are listed in the [requirements.txt](/doc/requirements.txt). Install it in the following way:

```
pip install -r requirements.txt
```

To build documentation open command line in the repo root, change working directory to `doc` folder and use `make` using `html` target

```bash
cd doc
gmake html
```

To open the documentation you've just built in a browser, use `doc/build/html/index.html` file.
