from datetime import datetime
from rethinkdb import RethinkDB
from beacon.adict import ADict


def non_eager(fn):
    def _decorate(*args, run=False, **kwargs):
        cmd = fn(*args, **kwargs)
        if run:
            cmd = cmd.run(args[0].rc)
        return cmd
    return _decorate


class DBTracker:
    def __init__(self, host='localhost', port='28015'):
        self._project_id = None
        self._cluster_id = None
        self._tracker_id = None
        self.host = host
        self.port = port
        self.config = None
        self.r = None
        self.rc = None

    @property
    def project_id(self):
        return self._project_id

    @property
    def cluster_id(self):
        return self._cluster_id

    def prepare(self, project_id, cluster_id='root'):
        self.r = RethinkDB()
        self.rc = self.r.connect(self.host, self.port)
        self._project_id = project_id
        self._cluster_id = cluster_id
        db_create_op = self.r.db_create(self._project_id)
        self.r.db_list().contains(self._project_id).not_().branch(db_create_op, None).run(self.rc)
        self.rc.use(self._project_id)
        table_create_op = [
            self.r.table_create(self._cluster_id),
            self.r.table(self._cluster_id).index_create('structural_hash'),
            self.r.table(self._cluster_id).index_wait('structural_hash')
        ]
        self.r.table_list().contains(self._cluster_id).not_().branch(table_create_op, None).run(self.rc)

    def disconnect(self):
        self.rc.close()
        self.rc = None

    def set_new_index(self, key):
        index_create_op = [
            self.r.table(self._cluster_id).index_create(key),
            self.r.table(self._cluster_id).index_wait(key)
        ]
        self.r.table(self._cluster_id).index_list().contains(key).not_().branch(index_create_op, None).run(self.rc)

    def start_new_tracker(self, config):
        if not isinstance(config, ADict):
            config = ADict(config)
        config.structural_hash = config.get_structural_hash()
        self.config = config
        self._tracker_id = self.r.table(self._cluster_id).insert(config.pydict()).run(self.rc)['generated_keys'][0]

    def start_from_tracker_id(self, tracker_id):
        if self.r.table(self._cluster_id).get(tracker_id).not_().run(self.rc):
            tracker_list = self.r.table(self._cluster_id).run(self.rc)
            raise RuntimeError(
                f'{tracker_id} does not exist in database; Current tracker list is:\n{"-"*30}'
                f'"\n".join({tracker_list})'
            )
        self._tracker_id = tracker_id

    def append(self, key, value):
        tracked_logs = self.r.table(self._cluster_id).get(self._tracker_id)
        tracked_logs.update(lambda row: {
            key: row[key].default([]).append(value),
            'modified_at': self.now()
        }).run(self.rc)

    def merge(self, key, values):
        tracked_logs = self.r.table(self._cluster_id).get(self._tracker_id)
        tracked_logs.update(lambda row: {
            key: row[key].default([]).union(values),
            'modified_at': self.now()
        }).run(self.rc)

    def get(self, key):
        return self.r.table(self._cluster_id).get(self._tracker_id)[key].run(self.rc)

    def find_similar_experiments(self, config):
        structural_hash = config.get_structural_hash()
        return self.find_by_structural_hash(structural_hash)

    def find_by_structural_hash(self, structural_hash):
        return self.r.table(self._cluster_id).get_all(structural_hash, index='structural_hash').run(self.rc)

    @non_eager
    def now(self):
        return self.r.now().in_timezone(datetime.utcnow().astimezone().isoformat()[-6:])
