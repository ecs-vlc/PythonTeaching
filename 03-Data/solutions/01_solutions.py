#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:59:44 2019

@author: gparkes
"""
import pandas as pd

tips = pd.read_csv("datasets/tips.csv")

def task_1():
    return tips.query("((total_bill > 25) | (tip > 4)) & (smoker == 'No') & (time == 'Dinner')")


def task_2():
    return tips[["total_bill", "tip"]].corr()


def task_3():
    return tips.groupby(["day", "sex"]).agg({"total_bill":"mean", "tip":"mean"})


def task_4():
    return tips.sort_values(by=["smoker","tip"], ascending=[False, False]).iloc[:10, :]


