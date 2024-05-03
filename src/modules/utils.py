from ..collectors import DataCollector

# Hook
def collect_post_backward_module_state(module, grad_input, grad_output):
    dc = DataCollector()
    dc.collect_parameters(module, 'post')
    
# Not a hook :)
def collect_init_module_state(module):
    dc = DataCollector()
    dc.collect_parameters(module, 'init')