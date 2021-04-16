embARC MLI Documentation
==================================================

The embARC MLI documentation can be built using [Sphynx](http://sphinx-doc.org/) together with the theme provided by [Read the Docs](https://readthedocs.org/)

To build the documentation you first need to install Python. See [this instruction](/examples/tutorial_emnist_tensorflow#install-python-and-create-a-virtual-environment) as one of the ways to do so.

Requirements for building the embARC documentation are listed in the [requirements.txt](/doc/requirements.txt). Install it in the following way:

```
pip install -r doc/requirements.txt
```

To build documentation open command line in the repo root, change working directory to `doc` folder and use `make` using `html` target

```bash
cd doc
gmake html
```

To open the documentation, open `doc/build/html/index.html` in a browser.
