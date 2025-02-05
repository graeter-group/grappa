import pytest

def pytest_configure(config):
    import dgl
    import torch
    import grappa
