#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:59:34 2019

@author: gparkes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools as it
import seaborn as sns
from scipy import stats

# task 1
# 1
births = pd.read_csv("datasets/births.csv")
# 2
births["decade"] = 10*(births["year"]//10)
# 3
piv = births.pivot_table("births",index="decade", columns="gender", aggfunc="sum")
# 4 plot
def plot_task1_piv():
    plt.semilogy(piv.index, piv["F"], 'x-')
    plt.semilogy(piv.index, piv["M"], 'o-')
    plt.show()

def plot_task1_piv_byyear():
    births.pivot_table("births", index="year", columns="gender", aggfunc="sum").plot()
    plt.ylabel("Total births per year")
    plt.show()

plot_task1_piv()
plot_task1_piv_byyear()

# task 2
# 1
planets = pd.read_csv("datasets/planets.csv")
# 2
print(planets.dropna().describe())
# 3
print(planets.groupby("method")["orbital_period"].median())
# 4
decade = (10*(planets["year"]//10)).astype(str) + "s"
decade.name = "decade"
# 5
print(planets.groupby(["method",decade])["number"].sum().unstack().fillna(0))
# 6
def plot_task2_planets():
    plt.scatter(planets.mass, planets.orbital_period, c=planets.distance)
    plt.colorbar()
    plt.xlabel("Mass")
    plt.ylabel("Orbital Period")
    plt.show()

plot_task2_planets()

# task 3
# 1
fmri = pd.read_csv("datasets/fmri.csv")
# 2
fmri_m = fmri.set_index(["timepoint","event"])
# 3
def plot_median_signal():
    fmri.groupby("timepoint").signal.agg(["median"]).plot()
    plt.ylabel("Median Signal")
    plt.show()

plot_median_signal()
# 4
filter_f = lambda f: (f["signal"].mean() < -.02) or (f["signal"].mean() > .02)

def plot_filtered_mean_signal():
    fmri.groupby("timepoint").filter(filter_f).groupby("timepoint").signal.mean().plot()
    plt.ylabel("Mean signal")
    plt.show()

plot_filtered_mean_signal()
# 5
fmri.event = fmri.event.astype("category")
fmri.region = fmri.region.astype("category")

# 6
def plot_seaborn_fmri():
    sns.catplot(x="timepoint", y="signal", hue="subject",
                col="event", row="region", data=fmri, kind="point", alpha=.5)
    plt.show()

plot_seaborn_fmri()

# task 4
# 1
life_exp = pd.read_csv("datasets/life_expectancy.csv",index_col=0)
country_pop = pd.read_csv("datasets/country_population.csv",index_col=0)
fert_rat = pd.read_csv("datasets/fertility_rate.csv", index_col=0)
print(life_exp.shape, country_pop.shape, fert_rat.shape)

# 2
remove_words = ["World", "Europe", "Asia", "Euro", "European Union", "situation",
                "HIPC", "Latin America", "Pacific", "demographic", "income",
                "UN", "classification", "Sahara", "Middle East", "Caribbean", "IDA"]
# update lists by concatenating huge OR statement and remove words that contain these segments.
fert_rat = fert_rat[~fert_rat.index.str.contains("|".join(remove_words))]
country_pop = country_pop[~country_pop.index.str.contains("|".join(remove_words))]
life_exp = life_exp[~life_exp.index.str.contains("|".join(remove_words))]

# 3
def compare_drop(*dfs):
    for d in dfs:
        print("Dataframe dim:{}; before {}, after {}".format(d.shape[1], d.shape[0], d.dropna().shape[0]))
    return

compare_drop(life_exp, country_pop, fert_rat)
life_exp.dropna(inplace=True)
country_pop.dropna(inplace=True)
fert_rat.dropna(inplace=True)

# 4
dsets = [life_exp, country_pop, fert_rat]
drop_cols = ["Country Code", "Indicator Name", "Indicator Code"]
years = life_exp.drop(drop_cols, axis=1).columns.tolist()
nind = pd.MultiIndex.from_tuples(it.chain(zip(["life_exp"]*len(years),years),
                                   zip(["population"]*len(years),years),
                                   zip(["fert_rat"]*len(years),years)), names=["datatype","years"])

dfdrop = lambda df: df.drop(drop_cols, axis=1)

df_all = pd.concat([dfdrop(df) for df in dsets], axis=1, sort=False)
df_all.columns = nind
df_all.index.name = "country"

# 5
pop_change = df_all["population"].pct_change(1,axis=1).multiply(100).drop("1960",axis=1)
pop_change.index.name = "country"

# 6

countries = {
    "Western Europe": ["United Kingdom", "France", "Germany", "Spain", "Italy"],
    "East Asia": ["China", "Japan", "India", "Korea, Rep.", "Mongolia"],
    "Sub-Saharan Africa": ["Nigeria", "South Africa", "Congo, Dem. Rep.", "Somalia", "Kenya"],
    "Middle East": ["Saudi Arabia", "Iran, Islamic Rep.", "Israel", "Syrian Arab Republic", "Jordan"]
}

def plot_5_countries_4_regions(X, yl):
    fig, axes = plt.subplots(ncols=4, figsize=(18, 5))
    for a, region in zip(axes, countries):
        for country in countries[region]:
            sns.lineplot(data=X.loc[country], label=country, ax=a)
        a.plot([0, 55], [0, 0], 'k--')
        a.legend()
        a.xaxis.set_major_locator(plt.MaxNLocator(10))
        a.set_xlabel("Date")
        a.set_ylabel(yl)
        a.set_title(region)

    plt.tight_layout()
    plt.show()

plot_5_countries_4_regions(pop_change, yl = "Percent yearly change in population (%)")

# task 5
# 1
# accrue data-subsets
a1 = pd.concat([df_all["fert_rat"].iloc[:,1:6],df_all["fert_rat"].iloc[:,-5:]], axis=1).reset_index().melt(id_vars=["country"], value_name="Fertility Rate")
a2 = pd.concat([pop_change.iloc[:,:5],pop_change.iloc[:,-5:]], axis=1).reset_index().melt(id_vars=["country"],value_name="Yearly Change in Population")
a3 = pd.concat([df_all["life_exp"].iloc[:,1:6],df_all["life_exp"].iloc[:,-5:]], axis=1).reset_index().melt(id_vars=["country"], value_name="Life Expectancy")
indexes = ["country","years"]
task5_1_long = pd.concat([a1.set_index(indexes), a2.set_index(indexes), a3.set_index(indexes)], axis=1, sort=True).reset_index()

def plot_task5_1():
    # plot
    g=sns.FacetGrid(data=task5_1_long, col="years", col_wrap=5)
    g.map(sns.scatterplot, "Fertility Rate", "Yearly Change in Population", "Life Expectancy")
    plt.show()

plot_task5_1()

# 2
def chunkIt(L, n):
    """ Yield successive n-sized chunks from L.
    """
    for i in range(0, len(L), n):
        yield L[i:i+n]

# calculate year chunks
chunks = list(chunkIt(df_all["life_exp"].columns,5))
chunk_names = ["{}-{}".format(c[0],c[-1]) for c in chunks]
# generate the multi index
grouped_idx = it.chain(zip(["life_exp"]*len(chunks),chunk_names),
                       zip(["population"]*len(chunks),chunk_names),
                       zip(["fert_rate"]*len(chunks),chunk_names))
midx_hd = pd.MultiIndex.from_tuples(grouped_idx, names=["datatype", "year_group"])

# calculate raw mean measurements and concat.
half_decade = []
for g in ["life_exp", "population", "fert_rat"]:
    for name, chunk in zip(chunk_names,chunks):
        hd = df_all[g][chunk].mean(axis=1)
        hd.name = name
        half_decade.append(hd)
Half_Decade_stats = pd.concat(half_decade,axis=1)
Half_Decade_stats.columns = midx_hd
print(Half_Decade_stats.head())

# 3
adjusted_fert = df_all["fert_rat"].sub(2.)
print(adjusted_fert.head(2))

# 4
plot_5_countries_4_regions(adjusted_fert, yl = "Adjusted Fertility Rate")

# task 6
# 1
sigma_v = np.linspace(2, 12, 100)
gamma_t = pd.Series(np.linspace(2, 4, df_all["life_exp"].columns.shape[0]),
                    index=df_all["life_exp"].columns)
scaler = 10
X = np.linspace(0, 100, 100*scaler)
age_bottom, age_top = 14, 38
age_diff = age_top - age_bottom

def new_babies(country, year, best_sigma):
    # calculate the number of women in the population
    n_women = df_all["population"].loc[country,year]*0.47
    # generate the pdf for the gamma distribution of women ages according to best features
    age_dist = stats.gamma.pdf(X, gamma_t[year], -1, best_sigma)
    # estimated breeding proportion is then the PDF sum scaling by the magnitude
    est_breeding_prop = np.sum(age_dist[age_bottom*scaler: age_top*scaler]) / scaler
    # return the estimated number of babies as the breeding proportion by women, scaled by fertile range,
    # and multiply by the fertility rate.
    return int(((n_women * est_breeding_prop) / age_diff) * df_all["fert_rat"].loc[country, year])

# 2
birth_rates = []
for year in years:
    sigma_r = np.asarray([stats.gamma.interval(.95, gamma_t[year], -1, s)[1] for s in sigma_v])
    # for each country in this year, calculate their best sigma.
    year_subset = df_all.dropna()["life_exp"][year]
    # get the best sigmas by choosing ones with minimum distance from our sigma set.
    best_sig = year_subset.apply(lambda x: sigma_v[np.argmin(np.abs(sigma_r - x))])
    # for each country
    birth_rate = pd.Series([new_babies(c, year, bs) for c, bs in best_sig.items()],
                            index=best_sig.index, name=year)
    birth_rates.append(birth_rate)

BR = pd.concat(birth_rates, axis=1)
print(BR.head())

estimated_pop_transf = BR.T.transform(lambda x: x.cumsum()) + df_all["population","1960"]
print(estimated_pop_transf.head())

# a function for factoring for plots - not essential
def factor(n):
    """
    Collect a list of factors given an integer, excluding 1 and n
    """
    if not isinstance(n, (int, np.int)):
        raise TypeError("'n' must be an integer")

    def prime_powers(n):
        # c goes through 2, 3, 5 then the infinite (6n+1, 6n+5) series
        for c in it.accumulate(it.chain([2, 1, 2], it.cycle([2,4]))):
            if c*c > n: break
            if n % c: continue
            d, p = (), c
            while not n % c:
                n, p, d = n // c, p * c, d + (p,)
            yield(d)
        if n > 1: yield((n,))

    r = [1]
    for e in prime_powers(n):
        r += [a*b for a in r for b in e]
    return r

def plot_country_pop(EST, ACT, countries=["Germany","United Kingdom","Saudi Arabia","China"]):
    n = len(countries)
    factors = np.asarray(factor(n))
    mncols = factors[len(factors)//2]
    mnrows = n // factors[len(factors)//2]
    # declare function
    fig, ax = plt.subplots(ncols=mncols, nrows=mnrows, figsize=(4*mncols,4*mnrows))
    # unwrap ax
    ax = list(it.chain.from_iterable(ax))
    for i,c in enumerate(countries):
        ax[i].plot(ACT.loc[c], 'x', label="real pop")
        ax[i].plot(EST.loc[c], 'x', label="est pop")
        ax[i].xaxis.set_major_locator(plt.MaxNLocator(7))
        ax[i].set_title(c)
        ax[i].legend()
    plt.tight_layout()
    plt.show()
    return ax

_=plot_country_pop(estimated_pop_transf.T,
                   df_all["population"],
                   df_all.dropna().sample(n=8).index)

# 3
# death rates per 1000
death_rat = pd.Series(np.linspace(19.1, 8.1, 57),
                      index=estimated_pop_transf.index).div(1000)

# estimate of dead population.
dead_pop = (df_all["population"]*death_rat).dropna().astype(np.int_)
# re-calculate estimate of population.
reestimated_pop_transf = BR.sub(dead_pop).dropna().T.transform(lambda x: x.cumsum()) + df_all["population","1960"]

_ = plot_country_pop(reestimated_pop_transf.T,
                     df_all["population"],
                     df_all.dropna().sample(n=8).index)