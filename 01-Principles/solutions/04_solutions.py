#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:27:52 2019

@author: gparkes
"""
import re

def task_1():
    strings = ["can", "man", "fan", "dan", "ran", "pan"]
    print([re.match("^[cmf]an", s) for s in strings])


def task_2():
    usernames = ["wA854k_12", "xQ764b-19", "oA488n_86", "vK221i_09"]
    print([re.findall("[a-z][A-Z]([0-9]{3})[a-z](?:\-|_)[0-9]{2}", u)[0] for u in usernames])


def task_3():
    URLs = ["https://www.google.co.uk","http://www.yahoo.com","www.wikipedia.org/wiki/Main_Page"]
    p = re.compile("(https?://www|www)(\.[a-z]+\.[a-z\.]{2,6}\/?[a-zA-Z_/]*)", re.IGNORECASE)
    print([p.match(url) for url in URLs])


def task_4():
    htmls = ['<img src="smileyface.gif" alt="Smiley Face" height="42" width="35">',
         '<img src="http://www.example.com/image.gif" alt="An example image" style="width:500px; height:600px;">']
    """
    Parameter 1: IMG
    2: ALT
    3: HEIGHT
    4: WIDTH
    5: STYLE
    """
    compile_str = '(?:<img src=\"(?P<IMG>[a-z0-9A-Z_\/:.]+\.[a-z]{1,3})\" alt=\"(?P<ALT>[a-zA-Z0-9\s._]+)\"' + \
    '(?:\sheight=\"(?P<HEIGHT>[0-9]+)?\")?(?:\swidth=\"(?P<WIDTH>[0-9]+)?\")?' + \
    '(?:\sstyle=\"(?P<STYLE>[a-zA-Z0-9:;\s]+)?\")?>)'
    p = re.compile(compile_str)
    print([p.findall(html) for html in htmls])
