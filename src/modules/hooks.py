from ..collectors import DataCollector

def collect_post_backward_module_state(module, grad_input, grad_output):
    dc = DataCollector()
    dc.collect_parameters(module, 'post-backward')

def collect_init_module_state_once(module, *args):
    dc = DataCollector()
    dc.collect_parameters(module, 'init')
    module.init_state_handle.remove()