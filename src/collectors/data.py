import namegenerator

from ..utils.singleton import SingletonMeta

class DataCollector(metaclass=SingletonMeta):
    def __init__(self):
        self.name = namegenerator.gen()
        self.current_epoch = None
        self.current_batch = None
        
        self.data = {}
        
    def collect_parameters(self, module, phase):
        if phase not in ['init', 'post']:
            raise ValueError('Invalid backward phase. Proper values are init or post.')
        
        module_name = module.__class__.__name__
        
        key = (self.current_epoch, self.current_batch, module_name, module.depth)
        
        if key not in self.data:
            self.data[key] = {}
            
        self.data[key][phase] = module.state_dict() 