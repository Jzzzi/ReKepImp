import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(__file__))
from utils.utils import get_config
from enviroment import RealEnviroment
from constraint_generator import ConstraintGenerator
from subgoal_solver import SubgoalSolver
from path_solver import PathSolver
# from visualizer import Visualizer

class Main:
    def __init__(self, visualize=False):
        self._global_config = get_config()
        self._main_config = self._global_config['main']
        self._bounds_min = np.array(self._global_config['bounds_min'])
        self._bounds_max = np.array(self._global_config['bounds_max'])
        self._visualize = visualize
        # set random seed
        np.random.seed(self._global_config['seed'])
        torch.manual_seed(self._global_config['seed'])
        torch.cuda.manual_seed(self._global_config['seed'])
        # constraint generator
        self._constraint_generator = ConstraintGenerator(self._global_config['constraint_generator'])
        # initialize environment
        self._env = RealEnviroment(self._global_config)

        # initialize solvers
        self._subgoal_solver = SubgoalSolver(self._global_config['subgoal_solver'])
        # TODO set the path solver
        self._path_solver = PathSolver()
        # initialize visualizer
        # if self._visualize:
        #     self._visualizer = Visualizer()

# for test
if __name__ == '__main__':
    main = Main()
