import os
import pickle
import namegenerator
from copy import deepcopy

from ..utils.singleton import SingletonMeta
from ..utils.distance import parameters_distance


class DataCollector(metaclass=SingletonMeta):
    def __init__(self, name=None, work_dir=None, metric_kwargs = {}):
        self.name = name or namegenerator.gen()
        self.work_dir = work_dir
        self.metric_kwargs = metric_kwargs

        self.current_epoch = None
        self.current_batch = None

        self.init_states = {}
        self.previous_states = {}
        self.next_differences = {}
        self.init_differences = {}

    def collect_parameters(self, module, phase):
        if phase not in ["init", "post-backward"]:
            raise ValueError("Invalid phase. Proper values are init or post-backward.")

        if phase == 'init':
            self.init_states[module.depth] = deepcopy(module.state_dict())
            return
            
        if module.depth not in self.previous_states:
            self.previous_states[module.depth] = deepcopy(module.state_dict())
            return
            
        init = self.init_states[module.depth]
        prev = self.previous_states[module.depth]
        curr = deepcopy(module.state_dict())
        
        self.previous_states[module.depth] = deepcopy(module.state_dict())
        
        if module.depth not in self.init_differences:
            self.init_differences[module.depth] = []
            
        if module.depth not in self.next_differences:
            self.next_differences[module.depth] = []
        
        self.init_differences[module.depth].append(parameters_distance(init, curr, **self.metric_kwargs))
        self.next_differences[module.depth].append(parameters_distance(prev, curr, **self.metric_kwargs))
        
    def save_state(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with open(os.path.join(directory, "init_states.pkl"), "wb+") as f:
            pickle.dump(self.init_states, f)
            
        with open(os.path.join(directory, "previous_states.pkl"), "wb+") as f:
            pickle.dump(self.previous_states, f)
            
        with open(os.path.join(directory, "init_differences.pkl"), "wb+") as f:
            pickle.dump(self.init_differences, f)
            
        with open(os.path.join(directory, "next_differences.pkl"), "wb+") as f:
            pickle.dump(self.next_differences, f)
            
