#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:22:49 2019

@author: gparkes
"""
import itertools as it

def task_1():
    def first_order(p, q, initial_val):
        return it.accumulate(it.repeat(initial_val), lambda s,_: p*s+q)


def task_2():
    def second_order(p, q, r, initial_values):
        intermediate = it.accumulate(it.repeat(initial_values), lambda s,_: (s[1], p*s[1] + q*s[0] + r))
        return map(lambda x: x[0], intermediate)


def task_3():
    bases = ['A', 'C', 'G', 'T']
    print(it.combinations_with_replacement(bases, 3))


def task_4():
    suits=["H","C","D","S"]
    ranks=["2","3","4","5","6","7","8","9","10","J","Q","K","A"]
    from sympy import factorial
    # count the proportion
    count_prop = len(list(it.filterfalse(lambda x: list(it.chain.from_iterable(x)).count("H") < 2,
                                         it.filterfalse(lambda x: ("Q","S") not in x, it.combinations(it.product(ranks, suits), 7)))))
    # we use sumpy to avoid floating point precision
    # using formula nCr = n! / r! * (n - r)!
    total_combs = lambda n ,r: factorial(n) / factorial(r) * factorial(n - r)
    total = total_combs(52, 7).evalf(20)
    print(count_prop / total)


def task_5():
    def frange(start,stop,increment=1):
        x = start
        while x < stop:
            yield x
            x += increment

    print(list(frange(0, 5, .5)))
