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

        if self.work_dir is not None:
            module_name = module.__class__.__name__
            key = (self.current_epoch, module_name, module.depth, self.current_batch)
            key_dir = os.path.join(self.work_dir, *map(str, key))

            if not os.path.exists(key_dir):
                os.makedirs(key_dir)

            with open(os.path.join(key_dir, f"{phase}.pkl"), "wb+") as f:
                pickle.dump(module.state_dict(), f)

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