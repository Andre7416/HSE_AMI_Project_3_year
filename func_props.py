from dataclasses import dataclass, field
from itertools import accumulate

import sympy as sp
from typing import List

t = sp.Symbol('t', real=True)
x = sp.Symbol('x', real=True)


@dataclass
class Func:
    func: sp.Expr
    check_interval: sp.Interval
    convexity_intervals: sp.Interval = sp.EmptySet
    concavity_intervals: sp.Interval = sp.EmptySet
    increasing_intervals: sp.Interval = sp.EmptySet
    decreasing_intervals: sp.Interval = sp.EmptySet
    positive_intervals: sp.Interval = sp.EmptySet
    negative_intervals: sp.Interval = sp.EmptySet
    possible_compositions: List[tuple] = field(default_factory=list)


data = dict()
pos = sp.EmptySet
neg = sp.EmptySet
incr = sp.EmptySet
decr = sp.EmptySet
for i in range(-4, 5):
    pos = sp.Interval.union(pos, sp.Interval(2 * sp.pi * i, 2 * sp.pi * i + sp.pi))
    neg = sp.Interval.union(neg, sp.Interval(2 * sp.pi * i + sp.pi, 2 * sp.pi * i + 2 * sp.pi))
    incr = sp.Interval.union(incr, sp.Interval(-sp.pi / 2 + 2 * sp.pi * i, 2 * sp.pi * i + sp.pi / 2))
    decr = sp.Interval.union(decr, sp.Interval(sp.pi / 2 + 2 * sp.pi * i, 2 * sp.pi * i + 3 * sp.pi / 2))

data[x] = Func(x, sp.Reals,
               increasing_intervals=sp.Reals,
               decreasing_intervals=sp.EmptySet,
               positive_intervals=sp.Interval(0, sp.oo, right_open=True),
               negative_intervals=sp.Interval(-sp.oo, 0, left_open=True))
data[x**2] = Func(x**2, sp.Reals,
                  increasing_intervals=sp.Interval(0, sp.oo, right_open=True),
                  decreasing_intervals=sp.Interval(-sp.oo, 0, left_open=True),
                  positive_intervals=sp.Reals,
                  negative_intervals=sp.EmptySet)
data[sp.sin(x)] = Func(sp.sin(x), sp.Reals,
                       increasing_intervals=incr,
                       decreasing_intervals=decr,
                       positive_intervals=pos,
                       negative_intervals=neg,
                       convexity_intervals=neg,
                       concavity_intervals=pos)
data[sp.cos(x)] = Func(sp.cos(x), sp.Reals,
                       increasing_intervals=neg,
                       decreasing_intervals=pos,
                       positive_intervals=incr,
                       negative_intervals=decr,
                       convexity_intervals=decr,
                       concavity_intervals=incr)

data[sp.Abs(x)] = Func(sp.Abs(x), sp.Reals,
                       increasing_intervals=sp.Interval(0, sp.oo, right_open=True),
                       decreasing_intervals=sp.Interval(-sp.oo, 0, left_open=True),
                       positive_intervals=sp.Reals,
                       negative_intervals=sp.EmptySet,
                       convexity_intervals=sp.Reals,
                       concavity_intervals=sp.EmptySet)
data[sp.exp(x)] = Func(sp.exp(x), sp.Reals,
                       increasing_intervals=sp.Reals,
                       decreasing_intervals=sp.EmptySet,
                       positive_intervals=sp.Reals,
                       negative_intervals=sp.EmptySet,
                       convexity_intervals=sp.Reals,
                       concavity_intervals=sp.EmptySet)
data[sp.log(x)] = Func(sp.log(x), sp.Interval(0, sp.oo, left_open=True, right_open=True),
                       increasing_intervals=sp.Interval(0, sp.oo, left_open=True, right_open=True),
                       decreasing_intervals=sp.EmptySet,
                       positive_intervals=sp.Interval(1, sp.oo, right_open=True),
                       negative_intervals=sp.Interval(0, 1, left_open=True),
                       convexity_intervals=sp.EmptySet,
                       concavity_intervals=sp.Interval(0, sp.oo, left_open=True, right_open=True))


class FuncProps:
    def __init__(self, func, interval):
        self.f = Func(func, interval)
        self.interval = interval

    def pre(self, expr, fu):
        for arg in expr.args:
            if arg.has(x) and not (arg / x).is_constant() and not (arg * x).is_constant():
                tmp = fu.func.subs(arg, t)
                if not tmp.has(x) and not (tmp / t).is_constant() and not (tmp * t).is_constant():
                    fu.possible_compositions.append((tmp.subs(t, x), arg))
                self.pre(arg, fu)

    def get_compositions(self, fu):
        self.pre(fu.func, fu)
        fu.possible_compositions = list(set(fu.possible_compositions))

    def diff_set(self, st, f):
        ret = sp.EmptySet
        period = sp.periodicity(f, x)
        if isinstance(st, sp.ConditionSet):
            return sp.EmptySet
        if isinstance(st, sp.Interval):
            ret += st
            if period is not None:
                for i in range(-4, 5):
                    ret += sp.Interval(st.left + period * i, st.right + period * i, left_open=st.left_open, right_open=st.right_open)
            return ret
        if isinstance(st, sp.Union):
            for st1 in st.args:
                if isinstance(st1, sp.Interval):
                    ret += st1
                    if period is not None:
                        for i in range(-4, 5):
                            ret += sp.Interval(st1.left + period * i,
                                               st1.right + period * i, left_open=st1.left_open,
                                               right_open=st1.right_open)
        return ret

    def lambda_mon_pow(self, a, b):
        inta1 = sp.Interval.intersect(a.increasing_intervals, a.positive_intervals)
        inta2 = sp.Interval.intersect(a.increasing_intervals, a.negative_intervals)
        inta3 = sp.Interval.intersect(a.decreasing_intervals, a.negative_intervals)
        inta4 = sp.Interval.intersect(a.decreasing_intervals, a.positive_intervals)
        intb1 = sp.Interval.intersect(b.increasing_intervals, b.positive_intervals)
        intb2 = sp.Interval.intersect(b.increasing_intervals, b.negative_intervals)
        intb3 = sp.Interval.intersect(b.decreasing_intervals, b.negative_intervals)
        intb4 = sp.Interval.intersect(b.decreasing_intervals, b.positive_intervals)

        incr = sp.Interval.union(sp.Interval.intersect(inta1, intb1), sp.Interval.intersect(inta3, intb3))
        incr2 = sp.Interval.union(sp.Interval.intersect(inta2, intb4), sp.Interval.intersect(inta4, intb2))
        incr = sp.Interval.union(incr, incr2)

        decr = sp.Interval.union(sp.Interval.intersect(inta4, intb4), sp.Interval.intersect(inta2, intb2))
        decr2 = sp.Interval.union(sp.Interval.intersect(inta3, intb1), sp.Interval.intersect(inta1, intb3))
        decr = sp.Interval.union(decr, decr2)

        pos = sp.Interval.union(sp.Interval.intersect(a.positive_intervals, b.positive_intervals), sp.Interval.intersect(a.negative_intervals, b.negative_intervals))

        neg = sp.Interval.union(sp.Interval.intersect(a.positive_intervals, b.negative_intervals), sp.Interval.intersect(a.negative_intervals, b.positive_intervals))

        return Func(a.func * b.func, a.check_interval, increasing_intervals=incr, decreasing_intervals=decr, positive_intervals=pos, negative_intervals=neg)

    def lambda_mon_sum(self, a, b):
        incr = sp.Interval.intersect(a.increasing_intervals, b.increasing_intervals)
        decr = sp.Interval.intersect(a.decreasing_intervals, b.decreasing_intervals)
        pos = sp.Interval.intersect(a.positive_intervals, b.positive_intervals)
        neg = sp.Interval.intersect(a.negative_intervals, b.negative_intervals)
        st1 = sp.solveset((a.func + b.func) >= 0, x, sp.Reals)
        st2 = sp.solveset((a.func + b.func) <= 0, x, sp.Reals)

        pos = sp.Interval.union(pos, self.diff_set(st1, a.func + b.func))

        neg = sp.Interval.union(neg, self.diff_set(st2, a.func + b.func))
        return Func(a.func + b.func, a.check_interval, increasing_intervals=incr, decreasing_intervals=decr,
                    positive_intervals=pos, negative_intervals=neg)

    def lambda_mon_comp(self, f1, f2):
        f1, f2 = Func(f1, self.interval), Func(f2, self.interval)
        f1 = self.get_monotonic(f1)
        f2 = self.get_monotonic(f2)

        incr = sp.EmptySet
        decr = sp.EmptySet
        if f1.increasing_intervals == sp.Reals:
            incr = incr + f2.increasing_intervals
            decr = decr + f2.decreasing_intervals
        if f1.decreasing_intervals == sp.Reals:
            decr = decr + f2.increasing_intervals
            incr = incr + f2.decreasing_intervals
        if f1.increasing_intervals.is_superset(sp.Interval(0, sp.oo, right_open=True)):
            incr = incr + (f2.positive_intervals & f2.increasing_intervals)
            decr = decr + (f2.positive_intervals & f2.decreasing_intervals)
        if f1.increasing_intervals.is_superset(sp.Interval(-sp.oo, 0, left_open=True)):
            incr = incr + (f2.negative_intervals & f2.increasing_intervals)
            decr = decr + (f2.negative_intervals & f2.decreasing_intervals)
        if f1.decreasing_intervals.is_superset(sp.Interval(0, sp.oo, right_open=True)):
            incr = incr + (f2.positive_intervals & f2.decreasing_intervals)
            decr = decr + (f2.positive_intervals & f2.increasing_intervals)
        if f1.decreasing_intervals.is_superset(sp.Interval(-sp.oo, 0, left_open=True)):
            incr = incr + (f2.negative_intervals & f2.decreasing_intervals)
            decr = decr + (f2.negative_intervals & f2.increasing_intervals)
        return Func(f1.func.subs(x, f2.func), self.interval, increasing_intervals=incr, decreasing_intervals=decr)

    def get_monotonic(self, f):
        if f.func in data.keys():
            f = data[f.func]
        st1 = sp.solveset(f.func.diff(x) >= 0, x, sp.Reals)
        st2 = sp.solveset(f.func.diff(x) <= 0, x, sp.Reals)
        st3 = sp.solveset(f.func >= 0, x, sp.Reals)
        st4 = sp.solveset(f.func <= 0, x, sp.Reals)

        f.increasing_intervals = sp.Interval.union(f.increasing_intervals, self.diff_set(st1, f.func))

        f.decreasing_intervals = sp.Interval.union(f.decreasing_intervals, self.diff_set(st2, f.func))

        f.positive_intervals = sp.Interval.union(f.positive_intervals, self.diff_set(st3, f.func))

        f.negative_intervals = sp.Interval.union(f.negative_intervals, self.diff_set(st4, f.func))

        if len(f.func.as_ordered_terms()) > 1:
            tmp = list(Func(q, self.f.check_interval) for q in f.func.as_ordered_terms())
            for i in range(len(tmp)):
                tmp[i] = self.get_monotonic(tmp[i])
            tmp_func = list(accumulate(tmp, func=self.lambda_mon_sum))[-1]
            f.positive_intervals = sp.Interval.union(f.positive_intervals, tmp_func.positive_intervals)
            f.negative_intervals = sp.Interval.union(f.negative_intervals, tmp_func.negative_intervals)
            f.increasing_intervals = sp.Interval.union(f.increasing_intervals, tmp_func.increasing_intervals)
            f.decreasing_intervals = sp.Interval.union(f.decreasing_intervals, tmp_func.decreasing_intervals)
        sign = 1
        if len(f.func.as_ordered_factors()) > 1:
            tmp = list(Func(q, self.f.check_interval) for q in f.func.as_ordered_factors())
            ttmp = []
            for i in range(len(tmp)):
                if tmp[i].func.is_constant():
                    if tmp[i].func.is_negative:
                        sign *= -1
                    continue
                ttmp.append(self.get_monotonic(tmp[i]))
            tmp_func = list(accumulate(ttmp, func=self.lambda_mon_pow))[-1]
            if sign == -1:
                tmp_func.increasing_intervals, tmp_func.decreasing_intervals = tmp_func.decreasing_intervals, tmp_func.increasing_intervals
                tmp_func.positive_intervals, tmp_func.negative_intervals = tmp_func.negative_intervals, tmp_func.positive_intervals

            f.positive_intervals = sp.Interval.union(f.positive_intervals, tmp_func.positive_intervals)
            f.negative_intervals = sp.Interval.union(f.negative_intervals, tmp_func.negative_intervals)
            f.increasing_intervals = sp.Interval.union(f.increasing_intervals, tmp_func.increasing_intervals)
            f.decreasing_intervals = sp.Interval.union(f.decreasing_intervals, tmp_func.decreasing_intervals)
        self.get_compositions(f)
        k = f.possible_compositions
        for (f1, f2) in k:
            f_tmp = self.lambda_mon_comp(f1, f2)
            f.increasing_intervals += f_tmp.increasing_intervals
            f.decreasing_intervals += f_tmp.decreasing_intervals
        data[f.func] = f
        return f

    def lambda_con_sum(self, a, b):
        incr = sp.Interval.intersect(a.convexity_intervals, b.convexity_intervals)
        decr = sp.Interval.intersect(a.concavity_intervals, b.concavity_intervals)
        f = a.func + b.func
        st1 = sp.solveset(f.diff(x).diff(x) >= 0, x, sp.Reals)
        st2 = sp.solveset(f.diff(x).diff(x) <= 0, x, sp.Reals)
        incr += self.diff_set(st1, f)
        decr += self.diff_set(st2, f)
        return Func(a.func + b.func, a.check_interval, convexity_intervals=incr, concavity_intervals=decr)

    def lambda_con_pow(self, a, b):
        conv_pos_a = sp.Interval.intersect(a.convexity_intervals, a.positive_intervals)
        conv_pos_b = sp.Interval.intersect(b.convexity_intervals, b.positive_intervals)
        conc_neg_a = sp.Interval.intersect(a.concavity_intervals, a.negative_intervals)
        conc_neg_b = sp.Interval.intersect(b.concavity_intervals, b.negative_intervals)
        conv_pos_incr_a = sp.Interval.intersect(conv_pos_a, a.increasing_intervals)
        conv_pos_decr_a = sp.Interval.intersect(conv_pos_a, a.decreasing_intervals)
        conc_neg_incr_a = sp.Interval.intersect(conc_neg_a, a.increasing_intervals)
        conc_neg_decr_a = sp.Interval.intersect(conc_neg_a, a.decreasing_intervals)
        conv_pos_incr_b = sp.Interval.intersect(conv_pos_b, b.increasing_intervals)
        conv_pos_decr_b = sp.Interval.intersect(conv_pos_b, b.decreasing_intervals)
        conc_neg_incr_b = sp.Interval.intersect(conc_neg_b, b.increasing_intervals)
        conc_neg_decr_b = sp.Interval.intersect(conc_neg_b, b.decreasing_intervals)
        conv_int = (conv_pos_incr_a & conv_pos_incr_b) ^ (conv_pos_decr_a & conv_pos_decr_b) ^ (conc_neg_incr_a & conc_neg_incr_b) ^ (conc_neg_decr_a & conc_neg_decr_b)
        conc_int = (conv_pos_incr_a & conc_neg_decr_b) ^ (conv_pos_decr_a & conc_neg_incr_b) ^ (conc_neg_incr_a & conv_pos_decr_b) ^ (conc_neg_decr_a & conv_pos_incr_b)
        f = a.func * b.func
        st1 = sp.solveset(f.diff(x).diff(x) >= 0, x, sp.Reals)
        st2 = sp.solveset(f.diff(x).diff(x) <= 0, x, sp.Reals)
        conv_int += self.diff_set(st1, f)
        conc_int += self.diff_set(st2, f)
        return Func(a.func * b.func, a.check_interval, convexity_intervals=conv_int, concavity_intervals=conc_int)

    def lambda_con_comp(self, f1, f2):
        f1, f2 = Func(f1, self.interval), Func(f2, self.interval)
        f1 = self.get_convexity(f1)
        f2 = self.get_convexity(f2)
        incr = sp.EmptySet
        decr = sp.EmptySet
        if f1.increasing_intervals & f1.convexity_intervals == sp.Reals:
            incr = incr + f2.convexity_intervals
        if f1.decreasing_intervals & f1.convexity_intervals == sp.Reals:
            incr = incr + f2.concavity_intervals
        if f1.increasing_intervals & f1.concavity_intervals == sp.Reals:
            decr = decr + f2.concavity_intervals
        if f1.decreasing_intervals & f1.concavity_intervals == sp.Reals:
            decr = decr + f2.convexity_intervals

        if (f1.increasing_intervals & f1.convexity_intervals).is_superset(sp.Interval(0, sp.oo, right_open=True)):
            incr = incr + (f2.positive_intervals & f2.convexity_intervals)
        if (f1.increasing_intervals & f1.convexity_intervals).is_superset(sp.Interval(-sp.oo, 0, left_open=True)):
            incr = incr + (f2.negative_intervals & f2.convexity_intervals)

        if (f1.decreasing_intervals & f1.convexity_intervals).is_superset(sp.Interval(0, sp.oo, right_open=True)):
            incr = incr + (f2.positive_intervals & f2.concavity_intervals)
        if (f1.decreasing_intervals & f1.convexity_intervals).is_superset(sp.Interval(-sp.oo, 0, left_open=True)):
            incr = incr + (f2.negative_intervals & f2.concavity_intervals)

        if (f1.increasing_intervals & f1.concavity_intervals).is_superset(sp.Interval(0, sp.oo, right_open=True)):
            decr = decr + (f2.positive_intervals & f2.concavity_intervals)
        if (f1.increasing_intervals & f1.convexity_intervals).is_superset(sp.Interval(-sp.oo, 0, left_open=True)):
            decr = decr + (f2.negative_intervals & f2.concavity_intervals)

        if (f1.decreasing_intervals & f1.concavity_intervals).is_superset(sp.Interval(0, sp.oo, right_open=True)):
            decr = decr + (f2.positive_intervals & f2.convexity_intervals)
        if (f1.decreasing_intervals & f1.concavity_intervals).is_superset(sp.Interval(-sp.oo, 0, left_open=True)):
            decr = decr + (f2.negative_intervals & f2.convexity_intervals)

        return Func(f1.func.subs(x, f2.func), self.interval, convexity_intervals=incr, concavity_intervals=decr)

    def get_convexity(self, f: Func):
        if f.func in data.keys():
            f = data[f.func]
        st1 = sp.solveset(f.func.diff(x).diff(x) >= 0, x, sp.Reals)
        st2 = sp.solveset(f.func.diff(x).diff(x) <= 0, x, sp.Reals)
        f.convexity_intervals = sp.Interval.union(f.convexity_intervals, self.diff_set(st1, f.func))

        f.concavity_intervals = sp.Interval.union(f.concavity_intervals, self.diff_set(st2, f.func))

        if len(f.func.as_ordered_terms()) > 1:
            tmp = list(Func(q, self.f.check_interval) for q in f.func.as_ordered_terms())
            for i in range(len(tmp)):
                tmp[i] = self.get_convexity(tmp[i])
            tmp_func = list(accumulate(tmp, func=self.lambda_con_sum))[-1]
            f.convexity_intervals = sp.Interval.union(f.convexity_intervals, tmp_func.convexity_intervals)
            f.concavity_intervals = sp.Interval.union(f.concavity_intervals, tmp_func.concavity_intervals)
        sign = 1
        if len(f.func.as_ordered_factors()) > 1:
            tmp = list(Func(q, self.f.check_interval) for q in f.func.as_ordered_factors())
            ttmp = []
            for i in range(len(tmp)):
                if tmp[i].func.is_constant():
                    if tmp[i].func.is_negative:
                        sign *= -1
                    continue
                ttmp.append(self.get_convexity(tmp[i]))
            tmp_func = list(accumulate(ttmp, func=self.lambda_con_pow))[-1]
            if sign == -1:
                tmp_func.convexity_intervals, tmp_func.concavity_intervals = tmp_func.concavity_intervals, tmp_func.convexity_intervals

            f.convexity_intervals = sp.Interval.union(f.convexity_intervals, tmp_func.convexity_intervals)
            f.concavity_intervals = sp.Interval.union(f.concavity_intervals, tmp_func.concavity_intervals)
        self.get_compositions(f)
        for (f1, f2) in f.possible_compositions:
            f_tmp = self.lambda_con_comp(f1, f2)
            f.convexity_intervals += f_tmp.convexity_intervals
            f.concavity_intervals += f_tmp.concavity_intervals
        data[f.func] = f
        return f

    def get_all_props(self):
        self.get_compositions(self.f)
        self.f = self.get_monotonic(self.f)
        self.f = self.get_convexity(self.f)
        self.f.increasing_intervals = sp.Interval.intersection(self.f.increasing_intervals, self.interval)
        self.f.decreasing_intervals = sp.Interval.intersection(self.f.decreasing_intervals, self.interval)
        self.f.convexity_intervals = sp.Interval.intersection(self.f.convexity_intervals, self.interval)
        self.f.concavity_intervals = sp.Interval.intersection(self.f.concavity_intervals, self.interval)
        self.f.positive_intervals = sp.Interval.intersection(self.f.positive_intervals, self.interval)
        self.f.negative_intervals = sp.Interval.intersection(self.f.negative_intervals, self.interval)

    def check(self):
        print(f"Func: {self.f.func}")
        print(f"Possible compositions: {self.f.possible_compositions}")
        print(f"Convexity intervals: {self.f.convexity_intervals}")
        print(f"Concavity intervals: {self.f.concavity_intervals}")
        print(f"Increasing intervals: {self.f.increasing_intervals}")
        print(f"Decreasing intervals: {self.f.decreasing_intervals}")
        print(f"Positive intervals: {self.f.positive_intervals}")
        print(f"Negative intervals: {self.f.negative_intervals}")
