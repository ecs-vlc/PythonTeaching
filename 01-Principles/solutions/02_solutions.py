#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:16:15 2019

@author: gparkes
"""

def task_1():
    with open("wolfcran.txt", "r") as f:
        # read in lines
        booklines = [line.strip() for line in f.readlines()]

    words = [[word.strip().lower() for word in line.split(" ")] for line in booklines]

    def extend_lists_into_one(all_words):
        # create a function to extend the list
        def list_extender(the_list, w):
            return the_list.extend(w)
        # iterate over all sets to extend into one
        neww = []
        for i in range(len(all_words)):
            list_extender(neww, all_words[i])
        return neww

    print(len(set(extend_lists_into_one(words))))


def task_2():
    x = [[j for j in range(6) if j % 2 == 0] for i in range(6) if i % 2 == 0]
    print(x)


def task_3():
    import math

    def log_sum_product(first, *vals):
        return math.exp(math.log(first) + math.fsum([math.log(x) for x in vals]))

    print(log_sum_product(0.5, 0.5, 0.2, .3, 0.1))


def task_4():
    import math
    def check_integral_size(func):
        def helper(f, a, b, N):
            if ((b - a) - (math.pi / 2)) < 1e-8 and type(N) is int:
                return func(f, a, b, N)
            else:
                raise Exception("Boundary size is not pi/2!")
        return helper

    f = lambda x: math.cos(x)

    @check_integral_size
    def trapz(f, a, b, N):
        """
        Calculates composite trapezoidal integration.

        Where f is a function passed, which accepts one parameter (x)
            a is the lower-bound
            b is the upper-bound
            N is the size

        Returns the Integral.
        """
        h = (b-a)/float(N)
        sum_y = 0
        x = a
        for i in range(N):
            x += h
            sum_y += f(x)
        sum_y += .5 * (f(a) + f(b))
        return sum_y*h

    print(trapz(f, 0, math.pi/2, 10000))
    print(trapz(f, 1, math.pi, 10000))
    print(trapz(f, 0, math.pi, 10000))
    print(trapz(f, math.pi/2, math.pi, 10000))
