import pandas as pd


__all__ = ['Model']


class Model:
    
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame):
        self.nodes = nodes
        self.edges = edges
    
    @classmethod
    def from_data(self, *datasets):
        self.datasets = {}
        for dataset in datasets:
            name = dataset.DEFAULT_NAME
            if name in self.datasets:
                raise ValueError('only one dataset if each type is allowed')
            self.datasets[name] = dataset
        return cls()
