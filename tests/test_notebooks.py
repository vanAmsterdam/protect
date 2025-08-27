# test notebooks
# tests/test_notebook.py
import nbformat

from nbconvert.preprocessors import ExecutePreprocessor
import os

def test_run_inference_demo():
    notebook_path = "tutorial/inference_demo.ipynb"
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
    parent_dir = os.path.dirname(notebook_path)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": parent_dir}})

