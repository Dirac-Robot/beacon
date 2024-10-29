import warnings

from beacon.adict import ADict
from beacon.tracker.db_tracker import DBTracker

try:
    from accelerate.tracking import GeneralTracker, on_main_process
    use_accelerate_tracker = True
except ImportError:
    warnings.warn('GeneralTracker cannot be imported, and it will be replaced by duck-typing.')
    use_accelerate_tracker = False


class DuckTypedRTTracker:
    name = 'DuckTypedRTTracker'
    requires_logging_directory = True

    def __init__(self, run_name, host='localhost', port='28015'):
        self.run_name = run_name
        self._tracker = DBTracker(host, port)
        self._tracker.prepare(run_name)

    def store_init_configuration(self, values):
        self._tracker.start_new_tracker(ADict(values))

    def log(self, values, step=None):
        if step is not None:
            values.update(step=step)
        self._tracker.append('log', values)

    @property
    def tracker(self):
        return self._tracker


class RTTracker(GeneralTracker, DuckTypedRTTracker):
    name = 'RTTracker'

    @on_main_process
    def __init__(self, run_name, host='localhost', port='28015'):
        GeneralTracker.__init__(self)
        DuckTypedRTTracker.__init__(self, run_name, host, port)
        self.run_name = run_name
        self._tracker = DBTracker(host, port)
        self._tracker.prepare(run_name)

    @on_main_process
    def store_init_configuration(self, values):
        DuckTypedRTTracker.store_init_configuration(self, values)

    @on_main_process
    def log(self, values, step=None):
        DuckTypedRTTracker.log(self, values, step)
