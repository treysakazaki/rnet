from abc import ABC
from typing import Tuple, List
import numpy as np
from qgis.core import Qgis, QgsTask, QgsMessageLog
from rnet.core.shortestpath import ShortestPathEngine
from rnet.core.patches import render_path
from rnet.utils import create_and_queue



class Chromosome(ABC):
    
    pass


class Route(Chromosome):
    '''
    Class for representing a route.
    
    Parameters
    ----------
    route : :class:`numpy.ndarray`, shape (M+2,)
        Route. M is the number of destinations. The route contains the start
        and goal nodes, therefore, it contains M+2 elements.
    order : :class:`numpy.ndarray`, shape (M,)
        Order. M is the number of destinations.
    
    Attributes
    ----------
    M : int
        Number of destinations.
    path : List[int]
        Initialized as None, updated by the :meth:`evaluate` method.
    cost : float
        Initialized as None, updated by the :meth:`evaluate` method.
    '''
    
    def __init__(self, route, order):
        self._route = route
        self._order = order
        self.M = len(order)
        self.path = None
        self.cost = None
    
    def __repr__(self):
        return f'Route(route={self._route}, order={self._order}, cost={self.cost})'
    
    def crossover(self, other, n) -> Tuple['Route', 'Route']:
        '''
        Crossover.
        
        Routes are combined via single-point crossover.
        
        Orders are combined via order-based crossover.
        
        Parameters
        ----------
        other : :class:`Route`
            Other route.
        n : int
            Number of genes to cross.
        
        Returns
        -------
        Tuple[:class:`Route`, :class:`Route`]
        '''
        route1, order1 = self._route, self._order
        route2, order2 = other._route, other._order
        
        x = np.random.choice(self.M + 2)
        route1, route2 = self.single_point_crossover(route1, route2, x)
        
        pos = np.random.choice(self.M, n, False)
        order1, order2 = self.order_based_crossover(order1, order2, pos)
        return Route(route1, order1), Route(route2, order2)
    
    @staticmethod
    def single_point_crossover(r1, r2, x):
        '''
        Parameters
        ----------
        r1, r2 : :class:`numpy.ndarray`, shape (M+2,)
            Routes to cross.
        x : int
            Crossover point.
        '''
        new_r1 = r1.copy()
        new_r2 = r2.copy()
        new_r1[x:] = r2[x:]
        new_r2[x:] = r1[x:]
        return new_r1, new_r2
    
    @staticmethod
    def order_based_crossover(o1, o2, pos):
        new_o1 = o1.copy()
        new_o2 = o2.copy()
        genes = o2[pos]
        for i, k in enumerate(np.argwhere(np.isin(o1, o2[pos]))):
            new_o1[k] = genes[i]
        genes = o1[pos]
        for i, k in enumerate(np.argwhere(np.isin(o2, o1[pos]))):
            new_o2[k] = genes[i]
        return new_o1, new_o2
    
    def evaluate(self, engine) -> None:
        '''
        Calculate route path and cost, then update the :attr:`path` and
        :attr:`cost` attributes.
        
        Parameters
        ----------
        engine : :class:`ShortestPathEngine`
            Engine for querying shortest paths.
        '''
        route = self.route
        total_path = [route[0]]
        total_cost = 0
        for k in range(self.M + 1):
            path, cost = engine.query(route[k], route[k+1])
            total_path.extend(path[1:])
            total_cost += cost
        self.path = total_path
        self.cost = total_cost
    
    @property
    def route(self) -> List[int]:
        '''List[int]: Route encoded by this chromosome.'''
        return np.hstack((self._route[0], self._route[self._order], self._route[-1]))


class Population:
    '''
    Parameters
    ----------
    routes : List[:class:`Route`]
        Routes. N is the population size. M is the number of destinations.
    
    Attributes
    ----------
    N : int
        Population size.
    '''
    
    def __init__(self, routes):
        self.routes = routes
        self.N = len(routes)
        self.costs = None
        self.selection = None
        self.evaluated = False
        self.selected = False
    
    def best(self):
        '''
        Returns the best chromosome in the population.
        
        Possible only after the population has been evaluated.
        
        Returns
        -------
        :class:`Route`
        '''
        assert self.evaluated
        return self.routes[np.argmin(self.costs)]
    
    def cross(self, *args, **kwargs):
        '''
        Returns child population after crossover operation.
        
        Selection must be completed first.
        '''
        assert self.selected
        routes = []
        for (p1, p2) in self.selection:
            routes.extend(self.routes[p1].crossover(self.routes[p2], *args, **kwargs))
        return Population(routes)
    
    def evaluate(self, engine):
        '''
        Evaluate chromosomes and update the :attr:`costs` attribute.
        '''
        costs = []
        for route in self.routes:
            route.evaluate(engine)
            costs.append(route.cost)
        self.costs = np.array(costs)
        self.evaluated = True
    
    @classmethod
    def init(self, pr, N):
        '''
        Randomly generate a population.
        
        Parameters
        ----------
        N : int
            Population size.
        '''
        S, G, D, bnodes, num_dsts = pr.S, pr.G, pr.D, pr.bnodes, pr.num_dsts
        
        routes = [np.full(N, S)]
        for d in D:
            routes.append(np.random.choice(bnodes[d], N))
        routes.append(np.full(N, G))
        routes = np.column_stack(routes)
        
        orders = []
        for _ in range(N):
            orders.append(np.random.permutation(num_dsts) + 1)
        orders = np.vstack(orders)
        
        routes = [Route(r, o) for (r, o) in zip(routes, orders)]
        return Population(routes)
    
    def mutate(self, pr, rho):
        '''
        Perform mutation at given rate.
        
        Parameters
        ----------
        rho : float
            Mutation rate.
        '''
        N = self.N
        P = np.random.rand(N)
        X = np.random.choice(pr.num_dsts, N) + 1
        for i in range(N):
            if P[i] < rho:
                x = X[i]
                current = self.routes[i]._route[x]
                d = pr.D[x-1]
                current_idx = np.flatnonzero(pr.bnodes[d]==current)[0]
                new_idx = np.mod(current_idx + np.random.choice([-1,1]),
                                 pr.num_bnodes[d])
                self.routes[i]._route[x] = pr.bnodes[d][new_idx]

    def select(self):
        '''
        Select via roulette-wheel selection and update the :attr:`selection`
        attribute.
        
        Selection is only possible after the population has been evaluated.
        '''
        assert self.evaluated
        f = 1 / self.costs
        p = f / np.sum(f)
        selection = []
        N = self.N
        for _ in range(int(N / 2)):
            selection.append(np.random.choice(N, 2, False, p))
        self.selection = np.vstack(selection)
        self.selected = True


class ProblemData:

    def __init__(self, graph_data, S, G, D):
        self.graph_data = graph_data
        self.S = S
        self.G = G
        self.D = D
        self.bnodes = graph_data.ndata.bnodes()
        self.num_dsts = len(self.D)
        self.edge_coords = graph_data.edge_coords()
        self.num_bnodes = {gr: len(bn) for gr, bn in self.bnodes.items()}


class ProblemSolver(QgsTask):
    '''
    Parameters
    ----------
    pr : :class:`ProblemData`
        Problem data.
    N : int
        Population size.
    max_iter : int
        Maximum number of iterations.
    rho : float
        Mutation rate.
    '''

    def __init__(self, pr, N, max_iter, rho):
        self.pr = pr
        self.N = N
        self.max_iter = max_iter
        self.rho = rho
        self.engine = ShortestPathEngine(pr.graph_data)
        super().__init__('Solving...')

    def run(self):
        # Unpacking
        pr, N, max_iter, rho = self.pr, self.N, self.max_iter, self.rho
        S, G, D = pr.S, pr.G, pr.D
        QgsMessageLog.logMessage(f'Running with S: {S}, G: {G}, D: {D}', 'test', Qgis.Info)

        # Initialize
        self.populations = {0: Population.init(pr, N)}
        self.populations[0].evaluate(self.engine)
        self.best_sol = (0, self.populations[0].best())
        QgsMessageLog.logMessage(f'best solution on iter 0: {self.best_sol}', 'test', Qgis.Info)

        # Algorithm
        i = 1
        while i < max_iter:
            # Selection
            self.populations[i-1].select()

            # Crossover
            self.populations[i] = self.populations[i-1].cross(2)

            # Mutation
            self.populations[i].mutate(pr, rho)
            
            # Evaluate
            self.populations[i].evaluate(self.engine)
            
            # Improved?
            best = self.populations[i].best()
            
            if best.cost < self.best_sol[1].cost:
                self.best_sol = (i, best)
                QgsMessageLog.logMessage(f'improved: {self.best_sol}', 'test', Qgis.Info)

            i += 1

        return True

    def finished(self, success: bool):
        if success:
            QgsMessageLog.logMessage(f'best solution: {self.best_sol}', 'test', Qgis.Info)
        else:
            QgsMessageLog.logMessage('failed', 'test', Qgis.Critical)
        
        route = self.best_sol[1]
        path = route.path
        coords = []
        for k in range(len(path) - 1):
            i, j = path[k], path[k+1]
            if i < j:
                coords.append(self.pr.edge_coords.loc[(i,j)][0])
            else:
                coords.append(self.pr.edge_coords.loc[(j,i)][0][::-1])
        coords = np.vstack(coords)
        render_path(coords)


def test(graphdata):
    global task
    areas = set(graphdata.ndata._df['gr'])
    areas.discard(-1)
    S, G = graphdata.rand(2)
    D = np.random.choice(list(areas), 4, replace=False)
#    S, G = 54517, 80277
#    D = [28, 49, 37]
    pr = ProblemData(graphdata, S, G, D)
    task = create_and_queue(ProblemSolver, pr, 50, 100, 0.5)
