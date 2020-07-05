import json

import networkx as nx
import numpy as np
import torch

from ratsql.datasets import spider
from ratsql.utils import registry
from third_party.wikisql.lib import dbengine
from third_party.wikisql.lib import query


def load_tables(paths):
    schemas = {}

    for path in paths:
        for line in open(path):
            schema_dict = json.loads(line)
            db_id = schema_dict['id']

            # Only one table in WikiSQL (without a real name)
            tables = (spider.Table(
                id=0,
                name=[db_id],
                unsplit_name=db_id,
                orig_name=db_id,
            ),)

            columns = tuple(
                spider.Column(
                    id=i,
                    table=tables[0],
                    name=col_name.split(),
                    unsplit_name=col_name,
                    orig_name=col_name,
                    type=col_type
                )
                for i, (col_name, col_type) in enumerate(zip(schema_dict['header'], schema_dict['types']))
            )

            # Link columns to tables
            for column in columns:
                if column.table:
                    column.table.columns.append(column)
            
            # No primary keys
            # No foreign keys
            foreign_key_graph = nx.DiGraph()
            # Final argument: don't keep the original schema
            schemas[db_id] = spider.Schema(db_id, tables, columns, foreign_key_graph, None)

    return schemas


@registry.register('dataset', 'wikisql')
class WikiSqlDataset(torch.utils.data.Dataset): 
    def __init__(self, paths, tables_paths, db_path, limit=None):
        self.paths = paths
        self.db_path = db_path
        self.examples = []
        self.schema_dicts = load_tables(tables_paths)

        for path in paths:
            for line in open(path):
                entry = json.loads(line)
                item = spider.SpiderItem(
                    text=entry['question'],
                    code=entry['sql'],
                    schema=self.schema_dicts[entry['table_id']],
                    orig={
                        'question': entry['question'],
                    },
                    orig_schema=None)
                self.examples.append(item)

                if limit and len(self.examples) > limit:
                    return

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    class Metrics:
        def __init__(self, dataset):
          self.dataset = dataset
          self.db_engine = dbengine.DBEngine(dataset.db_path)

          self.lf_match = []
          self.exec_match = []

        def _evaluate_one(self, item, inferred_code):
            gold_query = query.Query.from_dict(item.code, ordered=False)
            gold_result = self.db_engine.execute_query(item.schema.db_id, gold_query, lower=True)

            pred_query = None
            pred_result = None
            try:
                pred_query = query.Query.from_dict(inferred_code, ordered=False)
                pred_result = self.db_engine.execute_query(item.schema.db_id, pred_query, lower=True)
            except:
                # TODO: Use a less broad exception specifier
                pass

            lf_match = gold_query == pred_query
            exec_match = gold_result == pred_result

            return lf_match, exec_match

        def add(self, item, inferred_code, orig_question=None):
            lf_match, exec_match = self._evaluate_one(item, inferred_code)
            self.lf_match.append(lf_match)
            self.exec_match.append(exec_match)

        def finalize(self):
            mean_exec_match = float(np.mean(self.exec_match))
            mean_lf_match = float(np.mean(self.lf_match))

            return {
                'per_item': [{'ex': ex, 'lf': lf} for ex, lf in zip(self.exec_match, self.lf_match)],
                'total_scores': {'ex': mean_exec_match, 'lf': mean_lf_match},
            }
