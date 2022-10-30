import os
import pathlib
import time

import pandas as pd
import numpy as np

import streamlit as st
import sys

import nldbs
from datatype import DataType
from nlenv import NLEnv
from nlfilter import NLFilter
from query_info import NLQuery


class DummyDB():
    """ Represents ThalamusDB implementation. """

    def __init__(self, dbname):
        """ Initialize for given database.


        Args:
            dbname: name of database to initialize for
        """
        if dbname == 'YouTube':
            dbname = 'youtubeaudios'
        elif dbname == 'Craigslist':
            dbname = 'craigslist'
        self.nldb = nldbs.get_nldb_by_name(dbname)
        self.query = None
        self.error_constraint = None
        self.nl_filters = None

    def phase1(self, sql, error_constraint):
        """ Executes first phase of query processing: calculate
        similarity scores and identify objects to label.

        Args:
            query: an SQL query with natural language predicates
        """
        sql = sql.lower()
        self.query = NLQuery(sql)
        self.error_constraint = error_constraint
        self.nl_filters = [NLFilter(self.nldb.get_col_by_name(col), text) for col, text in self.query.nl_preds]

        query = self.query
        nl_filters = self.nl_filters

        # Create scores tables.
        for fid, nl_filter in enumerate(nl_filters):
            self.nldb.con.execute(f"DROP TABLE IF EXISTS scores{fid}")
            self.nldb.con.execute(f"CREATE TABLE scores{fid}(sid INTEGER PRIMARY KEY, score FLOAT, processed BOOLEAN)")
            if nl_filter.idx_to_score:
                self.nldb.con.execute(
                    f"INSERT INTO scores{fid} VALUES {', '.join(f'({key}, {val}, TRUE)' for key, val in nl_filter.idx_to_score.items())}")
            # self.nldb.con.executemany(f"INSERT INTO scores{fid} VALUES (?, ?, ?)",
            #                      [[key, val, True] for key, val in nl_filter.idx_to_score.items()])
            # Add null scores for remaining rows.
            self.nldb.con.execute(f"""
                    INSERT INTO scores{fid} SELECT {nl_filter.col.name}, NULL, FALSE
                    FROM (SELECT {nl_filter.col.name}, score FROM {nl_filter.col.table} LEFT JOIN scores{fid} ON {nl_filter.col.table}.{nl_filter.col.name} = scores{fid}.sid) AS temp_scores
                    WHERE score IS NULL""")
        # Get all possible orderings.
        possible_orderings = [('uniform',)]
        for col_name in query.cols:
            possible_orderings.append(('min', col_name))
            possible_orderings.append(('max', col_name))
        # Preprocess some data.
        preprocess_percents = [nl_filter.default_process_percent for nl_filter in nl_filters]
        fid2runtime = []
        for fid, nl_filter in enumerate(nl_filters):
            start_nl_filter = time.time()
            # To prevent bias towards uniform sampling.
            percent_per_ordering = preprocess_percents[fid] / len(possible_orderings)
            for ordering in possible_orderings:
                action = ('i', 1, fid) if ordering == ('uniform',) else (
                    'o', 1, query.cols.index(ordering[1]), ordering[0], fid)
                self.nldb.process_unstructured(action, query, nl_filters, percent_per_ordering)
            end_nl_filter = time.time()
            time_nl_filter = end_nl_filter - start_nl_filter
            print(f'Unit process runtime: {time_nl_filter}')
            fid2runtime.append(time_nl_filter)
        # # Collect 3 user feedbacks.
        # preprocess_nr_feedbacks = 5
        # for nl_filter in nl_filters:
        #     predicate = nl_filter.text
        #     for _ in range(preprocess_nr_feedbacks):
        #         target, idx = nl_filter.streamlit_collect_user_feedback_get(0.5)
        #         if nl_filter.col.datatype == DataType.AUDIO:
        #             st.audio(target)
        #         elif nl_filter.col.datatype == DataType.IMG:
        #             st.image(target)
        #         else:
        #             st.markdown(f'Text: {target}')
        #         label = st.radio(f'{predicate}?', options=['Yes', 'No'], index=1, key=(nl_filter.text, idx))
        #         is_yes = label == 'Yes'
        #         nl_filter.streamlit_collect_user_feedback_put(is_yes, idx)


@st.cache
def phase1(db, sql, error_constraint):
    print(f'Started Phase 1: {sql} {error_constraint}')
    db.phase1(sql, error_constraint)
    print(f'Finished Phase 1: {sql} {error_constraint}')


cur_file_dir = os.path.dirname(__file__)
src_dir = pathlib.Path(cur_file_dir).parent
root_dir = src_dir.parent
sys.path.append(str(src_dir))
sys.path.append(str(root_dir))
# print(f'sys.path: {sys.path}')
print('RELOADING')

os.environ['KMP_DUPLICATE_LIB_OK']='True'
st.set_page_config(page_title='ThalamusDB')
st.markdown('''
# ThalamusDB
ThalamusDB answers complex queries with natural 
language predicates on multi-modal data.
''')

with st.form("my_form"):
    with st.sidebar:
        metrics = ['Error', 'Computation', 'Interactions']
        constraint_on = st.selectbox('Constrained Metric:', options=metrics)
        constraint_max = 1.0 if constraint_on == 'Error' else 1000.0
        constraint = float(
            st.slider(
                f'Upper Bound ({constraint_on}):',
                min_value=0.0, max_value=constraint_max))

        weights = [-1] * 3
        for metric_idx, metric in enumerate(metrics):
            if not constraint_on == metric:
                weights[metric_idx] = st.slider(
                    f'Weight for {metric}:',
                    min_value=0.0, max_value=1000.0)

    database = st.selectbox('Database:', options=['Craigslist', 'YouTube'])
    sql = st.text_area('SQL Query with Natural Language Predicates:')

    # Every form must have a submit button.
    submitted = st.form_submit_button("Process Query")
    if submitted:
        st.write("Loading database...")
        db = DummyDB(database)

        st.write(f'Processing phase 1 ...')
        phase1(db, sql, constraint)
        st.session_state['label_idx'] = 0
        print(st.session_state['label_idx'])

        nl_filters = db.nl_filters
        # Collect 3 user feedbacks.
        preprocess_nr_feedbacks = 3
        for nl_filter in nl_filters:
            predicate = nl_filter.text
            for _ in range(preprocess_nr_feedbacks):
                target, idx = nl_filter.streamlit_collect_user_feedback_get(0.5)
                if nl_filter.col.datatype == DataType.AUDIO:
                    st.audio(target)
                elif nl_filter.col.datatype == DataType.IMG:
                    st.image(target)
                else:
                    st.markdown(f'Text: {target}')
                label = st.radio(f'{predicate}?', options=['Yes', 'No'], index=1, key=(nl_filter.text, idx))
                is_yes = label == 'Yes'
                nl_filter.streamlit_collect_user_feedback_put(is_yes, idx)

        st.write(f'Processing phase 2 ...')
        img_env = NLEnv(db.nldb, db.query, db.error_constraint, db.nl_filters)
        error, result_l, result_u = img_env.error_and_query_result()
        # result = (result_l, result_u)

        chart_data = pd.DataFrame([['Lower Bound', result_l], ['Upper Bound', result_u]], columns=["labels", "bounds"])
        chart_data = chart_data.set_index('labels')
        print(chart_data)

        st.bar_chart(chart_data)







