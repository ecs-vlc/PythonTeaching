#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:50:13 2019

@author: gparkes
"""

def task_1():
    for i in range(200):
        if i % 3 == 0 and i % 6 > 0:
            print(i)


def task_2():
    def factorial(n):
        a = 1
        for i in range(1, n):
            a *= i
        return a


def task_3():
    accounts = [
        ("John", "Doe", 23.50),
        ("Michael", "Mickey", 45.21),
        ("Sarah", "Wollaston", 32.0000000001),
        ("Ashley", "Carper", 12.06),
        ("Ferdinand", "Cortez", 80.75)
    ]

    for first, last, num in accounts:
        print("Hello {} {}, Your current balance is ${}".format(first, last, num))


def task_4():
    import random

    class Cipher(object):

        def _create_key(self):
            letters = list("abcdefghijklmnopqrstuvwxyz")
            # shuffle the letters
            random.shuffle(letters)
            # map the top half to the bottom half
            self.key = dict(zip(letters[:13],letters[13:]))
            # add reverse dictionary.
            self.key.update({v: k for k, v in self.key.items()})

        def __init__(self, key={}):
            # create a random key if not defined, else use the one passed
            if len(key)==0:
                self._create_key()
            else:
                self.key = key

        def _convert(self, s):
            # map each letter using the same key.
            return "".join([self.key[L] for L in list(s)])

        def encrypt(self, s):
            # since the key is symmetric, the encrypt and decrypt functions are the same.
            return self._convert(s)

        def decrypt(self, s):
            # since the key is symmetric, the encrypt and decrypt functions are the same.
            return self._convert(s)

        def set_code(self, K):
            if len(K) > 0 and isinstance(K, dict):
                self.key = K


def task_5():
    import math

    def fib_closed(n):
        return ((1+math.sqrt(5))**n - (1-math.sqrt(5))**n) / ((2**n) * math.sqrt(5))

    def fib_method(n):
        if n==0:
            return 0
        elif n == 1:
            return 1
        else:
            return fib_method(n-1) + fib_method(n-2)

    n = 20
    closed_l = [fib_closed(i) for i in range(n)]
    method_l = [fib_method(i) for i in range(n)]

    print(closed_l)
    print(method_l)
