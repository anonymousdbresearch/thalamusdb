import math
import re

from sqlglot import parse_one, exp


class NLQuery:
    """Query object with support for natural language predicates on unstructured data.

    RL-based optimizer takes this object as input.
    """

    def __init__(self, sql):
        self.sql = sql
        # Finds case-insensitive matches and extract arguments as tuples, e.g., [(col_name, nl_text), ...].
        self.arg_strs = [arg_str[1:] for arg_str in re.findall(r'(?i)[ \(]nl\(.+?, .+?\)', sql)]
        self.nl_preds = [tuple(arg.replace("'", '').replace('"', '').strip() for arg in arg_str[3:-1].split(',')) for
                         arg_str in self.arg_strs]
        # Find all tables and columns.
        self.parsed = parse_one(sql)
        self.cols = sorted(set([col.alias_or_name for col in self.parsed.find_all(exp.Column)]))
        self.tables = sorted(set([table.name for table in self.parsed.find_all(exp.Table)]))
        print(f'Columns: {self.cols}, Tables: {self.tables}')
        # Check if query has limit.
        limit_exps = list(self.parsed.find_all(exp.Limit))
        assert len(limit_exps) <= 1
        self.limit = -1 if len(limit_exps) == 0 else int(limit_exps[0].args['expression'].this)

    def get_table_by_col(self, col_name, nldb):
        for table_name in self.tables:
            if col_name in nldb.tables[table_name].cols:
                return table_name
        raise ValueError(f'No table with such column: {col_name}')

    def to_lower_upper_sqls(self, nl_filters, thresholds):
        assert len(nl_filters) >= 1
        # Replace nl predicates with predicates on similarity scores.
        sql_l, sql_u = self.sql, self.sql
        join_conditions = []
        for fid, nl_filter in enumerate(nl_filters):
            lower, upper = thresholds[fid]
            predicate_l = f"scores{fid}.score >= {upper}"
            predicate_u = f"(scores{fid}.score IS NULL OR scores{fid}.score > {lower})"
            prev_str = self.arg_strs[fid]
            sql_l = sql_l.replace(prev_str, predicate_l)
            sql_u = sql_u.replace(prev_str, predicate_u)
            join_conditions.append(f"scores{fid}.sid = {nl_filter.col.name}")
        # Add scores tables.
        join_condition = ' AND '.join(join_conditions)
        scores_tables = [f'scores{fid}' for fid in range(len(nl_filters))]
        parsed_l = parse_one(sql_l).from_(*scores_tables).where(join_condition)
        parsed_u = parse_one(sql_u).from_(*scores_tables).where(join_condition)
        # Add sum and count if there is avg.
        avgs = [select for select in self.parsed.selects if type(select) is exp.Avg]
        if len(avgs) > 0:
            for avg in avgs:
                col_name = avg.this.alias_or_name
                parsed_l = parsed_l.select(f'sum({col_name})').select(f'count({col_name})')
                parsed_u = parsed_u.select(f'sum({col_name})').select(f'count({col_name})')
        sql_l = parsed_l.sql()
        sql_u = parsed_u.sql()
        return sql_l, sql_u, len(avgs)

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class NLQueryInfo:
    """Query information for cost-based optimization.

    Currently, supports star schema.
    """

    def __init__(self, query):
        self.query = query
        # Find relevant query components.
        assert sum(1 for _ in query.parsed.find_all(exp.Select)) == 1, 'Subquery not yet supported.'
        assert type(query.parsed) is exp.Select, f'Query should start with SELECT: {type(query.parsed)}.'
        # Collect aggregates.
        assert len(query.parsed.selects) == 1
        agg = query.parsed.selects[0]
        if agg.alias_or_name == '*':
            self.agg_func = None
            self.agg_col = '*'
        else:
            self.agg_func = agg.key
            self.agg_col = agg.this.alias_or_name
        # Collect predicates.
        wheres = list(query.parsed.find_all(exp.Where))
        assert len(wheres) <= 1
        self.where = None if len(wheres) == 0 else wheres[0].this
        # Limit info is already in self.query.

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class TFUCardinality:
    """ Ternary (+ processed) cardinality.

    The semantics of unsure is different from counts of true, false, and unsure.
    That is, here, unsure includes unprocessed while, for counts, unsure only
    refers to processed items."""
    def __init__(self, s_true, s_false, s_unsure, p_true, p_false, p_unsure, nr_total, ordering_to_ratio):
        self.s_t = s_true
        self.s_f = s_false
        self.s_u = s_unsure
        self.p_t = p_true
        self.p_f = p_false
        self.p_u = p_unsure
        self.nr_total = nr_total
        self.ordering_to_ratio = ordering_to_ratio

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)

    @property
    def s_tu(self):
        return self.s_t + self.s_u

    @property
    def s_fu(self):
        return self.s_f + self.s_u

    @property
    def p_tu(self):
        return self.p_t + self.p_u

    @property
    def p_fu(self):
        return self.p_f + self.p_u

    def not_eval(self):
        return TFUCardinality(self.s_f, self.s_t, self.s_u, self.f, self.t, self.u, self.nr_total, self.ordering_to_ratio)

    def composite_ordering_to_ratio(self, other):
        if self.ordering_to_ratio is None:
            return other.ordering_to_ratio
        elif other.ordering_to_ratio is None:
            return self.ordering_to_ratio
        else:
            key_intersection = self.ordering_to_ratio.keys() & other.ordering_to_ratio.keys()
            return {key: min(self.ordering_to_ratio[key], other.ordering_to_ratio[key]) for key in key_intersection}

    def and_correlated(self, other, p_p):
        p_true = (self.s_t * other.s_t) * p_p
        p_false = (self.s_f + other.s_f - self.s_f * other.s_f) * p_p
        p_unsure = (self.s_u * other.s_u + self.s_t * other.s_u + self.s_u * other.s_t) * p_p
        return p_true, p_false, p_unsure

    def and_independent(self, other, p_p):
        p_remaining = 1 - p_p
        p_true = (self.p_t * other.p_t) * p_remaining
        p_false = (self.p_f + other.p_f - self.p_f * other.p_f) * p_remaining
        p_unsure = (self.p_u * other.p_u + self.p_t * other.p_u + self.p_u * other.p_t) * p_remaining
        return p_true, p_false, p_unsure

    def and_eval(self, other):
        ordering_to_ratio = self.composite_ordering_to_ratio(other)
        ratio_sum = sum(ordering_to_ratio.values())
        # Same ordering: so.
        p_true_so, p_false_so, p_unsure_so = self.and_correlated(other, ratio_sum)
        assert math.isclose(sum([p_true_so, p_false_so, p_unsure_so]), ratio_sum), f'{p_true_so}, {p_false_so}, {p_unsure_so}, {sum([p_true_so, p_false_so, p_unsure_so])}, {ratio_sum}'
        # Remaining: r.
        p_true_r, p_false_r, p_unsure_r = self.and_independent(other, ratio_sum)
        assert math.isclose(sum([p_true_r, p_false_r, p_unsure_r]), 1 - ratio_sum), f'{p_true_r}, {p_false_r}, {p_unsure_r}, {sum([p_true_r, p_false_r, p_unsure_r])}, {1 - ratio_sum}'
        # Overall.
        s_true = self.s_t * other.s_t
        s_false = self.s_f + other.s_f - self.s_f * other.s_f
        s_unsure = self.s_tu * other.s_tu - s_true
        assert math.isclose(sum([s_true, s_false, s_unsure]), 1), f'{s_true}, {s_false}, {s_unsure}, {sum([s_true, s_false, s_unsure])}'
        p_true = p_true_so + p_true_r
        p_false = p_false_so + p_false_r
        p_unsure = p_unsure_so + p_unsure_r
        assert math.isclose(sum([p_true, p_false, p_unsure]), 1), f'{p_true,}, {p_false}, {p_unsure}, {sum([p_true, p_false, p_unsure])}'
        return TFUCardinality(s_true, s_false, s_unsure, p_true, p_false, p_unsure, self.nr_total, ordering_to_ratio)

    def or_eval(self, other):
        ordering_to_ratio = self.composite_ordering_to_ratio(other)
        s_false = self.s_f * other.s_f
        s_unsure = self.s_fu * other.s_fu - s_false
        s_true = 1 - s_false - s_unsure
        p_false = self.p_f * other.p_f
        p_unsure = self.p_fu * other.p_fu - p_false
        p_true = 1 - p_false - p_unsure
        return TFUCardinality(s_true, s_false, s_unsure, p_true, p_false, p_unsure, self.nr_total, ordering_to_ratio)


class CardinalityModel:
    """Cardinality model for ternary (i.e., true, false, unsure) predicate evaluation.

    We need to create a new object for each specific query."""

    def __init__(self, query_info, max_nr_total):
        self.node = query_info.where
        self.query = query_info.query
        self.max_nr_total = max_nr_total
        self.f_nrss = None

    def eval(self, f_nrss):
        self.f_nrss = f_nrss
        return self._eval(self.node)

    @staticmethod
    def _is_nl_pred(node):
        return type(node) is exp.Anonymous and node.this.lower() == 'nl'

    def _get_fid(self, node):
        nl_filter_sql_str = str(node).lower()
        fid = self.query.arg_strs.index(nl_filter_sql_str)
        return fid

    def _eval(self, node):
        """Evaluates cardinality for the given predicate node."""
        node_type = type(node)
        if node_type is exp.Paren:
            # Parenthesis.
            return self._eval(node.this)
        elif node_type is exp.Not:
            return self._eval(node.this).not_eval()
        elif node_type is exp.And:
            cardi_left = self._eval(node.left)
            cardi_right = self._eval(node.right)
            return cardi_left.and_eval(cardi_right)
        elif node_type is exp.Or:
            cardi_left = self._eval(node.left)
            cardi_right = self._eval(node.right)
            return cardi_left.or_eval(cardi_right)
        elif node_type is exp.Anonymous:
            # Check keyword for natural language predicate.
            if not CardinalityModel._is_nl_pred(node):
                raise ValueError(f'Unknown user-defined function: {node.this}.')
            # TODO: Better way to refer to NLFilter.
            # Get ternary cardinality.
            fid = self._get_fid(node)
            nrs = self.f_nrss[fid]
            s_true = nrs.t / nrs.processed
            s_false = nrs.f / nrs.processed
            s_unsure = nrs.u / nrs.processed
            p_true = nrs.t / nrs.nr_total
            p_false = nrs.f / nrs.nr_total
            p_unsure = 1 - p_true - p_false
            ordering_to_ratio = {k: v / nrs.nr_total for k, v in nrs.ordering_to_cnt.items()}
            return TFUCardinality(s_true, s_false, s_unsure, p_true, p_false, p_unsure, self.max_nr_total, ordering_to_ratio)
        # TODO: Support more predicate types.
        elif node_type is exp.EQ:
            return TFUCardinality(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.max_nr_total, None)
        elif node_type is exp.LT or node_type is exp.LTE:
            return TFUCardinality(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.max_nr_total, None)
        elif node_type is exp.GT or node_type is exp.GTE:
            return TFUCardinality(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.max_nr_total, None)
        else:
            raise Exception(f'Unsupported predicate type: {node_type}, {node}.')


if __name__ == "__main__":
    sql = "select * from images, furniture where images.aid = furniture.aid and nl(img, 'blue chair') and nl(title_u, 'good condition') limit 10"
    query = NLQuery(sql)




