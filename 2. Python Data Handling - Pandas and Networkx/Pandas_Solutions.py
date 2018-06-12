#03-01

%%cython --annotate

import cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float64_t, ndim=1] gradient_descent2(np.ndarray[np.float64_t, ndim=2] X,
                     np.ndarray[np.float64_t, ndim=1] y,
                     double gamma = 1., 
                     int n_iter = 10**3):
    
    cdef:
        int i, j, k
        Py_ssize_t n, P
        double suma
        
    n = X.shape[0]
    P = X.shape[1]
    
    cdef np.ndarray[np.float64_t, ndim=2] nX = np.column_stack((X, np.ones(n)))
    cdef np.ndarray[np.float64_t, ndim=1] w = np.random.rand(P+1)
    cdef np.ndarray[np.float64_t, ndim=2] nXT = nX.T
    cdef np.ndarray[np.float64_t, ndim=1] tmp_d = np.empty(n)
    cdef np.ndarray[np.float64_t, ndim=1] tmp_dE = np.empty(P+1)
    
    # firstly calculate 2*X.T as we can take this out of the loop - it does not update
    for j in range(P):
        for k in range(n):
            nXT[k,j] *= 2.0

    for i in range(n_iter):
        # calculate dot(nX, w) - y
        for j in range(n):
            suma = 0
            for k in range(P+1):
                suma += (nX[k,j] * w[k])
            #tmp_d[j] = row_dot(nX,w,j) - y[j]
            tmp_d[j] = (suma - y[j])
        # calculate dot(2*nX.T, dot(nX, w) - y)
        for j in range(P+1):
            suma = 0
            for k in range(n):
                suma += (nXT[k,j] * tmp_d[k])
            #tmp_dE[j] = row_dot(nXT, tmp_d, j)
            tmp_dE[j] = suma
        # modify w by gamma * dE
        for j in range(P+1):
            w[j] -= (gamma * tmp_dE[j])
    return w


X = np.random.normal(3.0, 1.0, size=(10000,500))
y = np.random.randn(10000)
%timeit gradient_descent2(X, y, n_iter=300)


# 04-01

import numpy as np
import pandas as pd
G = nx.read_gml("GGI.gml")

edges, weights = zip(*nx.get_edge_attributes(G, "Weight").items())
weights = np.log(np.array(weights))

fig = plt.figure(figsize=(12,10))
nx.draw_circular(G, node_color='b', alpha=.5, node_size=20, edge_color=weights, edge_cmap=plt.cm.BuPu_r)

# 05-01

battles = pd.read_excel("battles.xlsx").drop('location', axis=1).set_index("battle_number")
battles.battle_type.value_counts().plot.pie()
battles.year.value_counts().plot.pie()
battles.region.value_counts().plot.pie()

# 05-02

pd.concat([
    battles.groupby(['attacker_king']).attacker_size.mean(),
    battles.groupby(['defender_king']).defender_size.mean()
], axis=1).mean(axis=1).idxmax()

# 05-03

battles = (pd.concat([
    battles,
    battles.attackers.str.split(', ', expand=True).apply(lambda x: x.str.strip()).add_prefix("attacker_"),
    battles.defenders.str.split(", ", expand=True).apply(lambda x: x.str.strip()).add_prefix("defender_"),
    battles.attacker_commander.str.split(',',expand=True).apply(lambda x: x.str.strip()).add_prefix("att_commander_"),
    battles.defender_commander.str.split(",",expand=True).apply(lambda x: x.str.strip()).add_prefix("def_commander_")
], axis=1).drop(["attackers","defenders","attacker_commander","defender_commander"], axis=1)
    .dropna(axis=0, subset=['attacker_king','defender_king','attacker_outcome']))
battles.head()

# 05-04

# extract all unique commanders
commanders = pd.DataFrame(battles.loc[:,"att_commander_0":"def_commander_6"].stack().unique(), columns=['Name'])
# guess house based on surname
commanders['House'] = commanders.Name.str.rstrip().str.lstrip().str.extract("\s([a-zA-Z]+)",expand=True)

# get allegiance to one of the kings
king_to_name = pd.concat([
    
    ((battles.loc[:,'att_commander_0':'att_commander_5']).join(battles.attacker_king)
    .melt(id_vars="attacker_king", value_name="Name")
    .dropna(subset=['Name']).drop('variable', axis=1).rename(columns={'attacker_king':'King'})),
    
    ((battles.loc[:,'def_commander_0':'def_commander_6']).join(battles.defender_king)
    .melt(id_vars="defender_king", value_name="Name")
    .dropna(subset=['Name']).drop('variable', axis=1).rename(columns={'defender_king':'King'}))
    
], axis=0).drop_duplicates()

def extract_king(row):
    # extract rows where the commander is present
    kings_for_name = king_to_name[king_to_name.Name.str.contains(row.Name)]
    # if there is only one king to the commander, take his name
    if len(kings_for_name) < 2:
        return kings_for_name.King.iloc[0]
    else:
        # attempt to eliminate duplicates
        attempt = kings_for_name.drop_duplicates(subset="King")
        if len(attempt) < 2:
            return attempt.King.iloc[0]
        else:
            # we genuinely have two different kings for the same name, cat them together
            return kings_for_name.King.str.cat(sep=";")

commanders['King'] = commanders.apply(extract_king, axis=1)
# major captures and deaths associated to - we only count it if they are associated on the *winning side* of the battle
comm_capt_death = (battles.loc[:,'att_commander_0':'att_commander_5'].join(battles[['major_death','major_capture']])
    .melt(id_vars=["major_death","major_capture"], value_name="Name")
    .groupby(["Name"]).sum().rename(columns={"major_death":'nMajor_Deaths',"major_capture":"nMajor_Captures"})
    .reset_index())
# merge together on name, and fill missing with 0
commanders = commanders.merge(comm_capt_death, how='left', on="Name").fillna(0)
# size of armies they controlled - we will take the median to eliminate skew
commanders = commanders.join((battles.loc[:,'att_commander_0':'def_commander_6']
    .join(battles[['attacker_size','defender_size']])
    .melt(id_vars=["attacker_size",'defender_size'], value_name="Name")
    .groupby("Name").median()
    .mean(axis=1).dropna().rename(index="Median_army_size")), on="Name", how="left")

# preferred type of battle - take the max with respect to each commander
commanders = commanders.join(
    (battles.loc[:,'att_commander_0':'att_commander_5'].join(battles.battle_type)
    .melt("battle_type", value_name="Name")
    .pivot(columns="battle_type", values="Name")
    .apply(lambda x: x.value_counts())
    .fillna(0).idxmax(axis=1).rename(index="Preferred_battle_type")), on="Name", how="left"
)
# number of victories and defeats associated with each commander
commanders = commanders.join(
    (battles.loc[:,'att_commander_0':'def_commander_6'].join(battles.attacker_outcome)
    .melt("attacker_outcome", value_name="Name")
    .pivot(columns="attacker_outcome", values="Name")
    .apply(lambda x: x.value_counts())
    .fillna(0).rename(columns={'loss':'nLosses', 'win':'nWins'})),
    on="Name", how="left"
)

# 05-05

chars = pd.read_excel("character-predictions.xlsx")
# extract targaryens
targaryens = (chars[chars.name.str.contains("Targaryen") | chars.house.str.contains("Targaryen")]
    .drop(["book1","book2","book3","book4","book5","culture","DateoFdeath","S.No",
            "isAliveMother","isAliveFather","isAliveHeir"],axis=1)
        .assign(house="House Targaryen"))

# attempt to extract (son of) or (daughter of) from name.
relations_list = targaryens.name.str.extract("[a-zA-Z]+\s[a-zA-Z]+\s\(([a-zA-Z\s]+)\)", expand=False).dropna()
# extract relation
relation = relations_list.str.extract("[a-z]+\sof\s(?P<father>[a-zA-Z\s]+)", expand=True)
# add to targaryens
targaryens['Father'] = relation.father.add(" Targaryen")
# shorten main name

# create nodes by using names + ones found in spouses
nodes = targaryens.name.values

# now try to build the network map, encompassing spouses and fathers
edges = pd.concat([
 # connections default found in the dataset
    (targaryens[['name','mother','father','heir']]
    .dropna(subset=['mother','father','heir'])
    .melt("name", var_name="Relation",value_name="Name2")),
    
    # spousal connections
    (targaryens[['name','spouse']]
     .dropna(subset=['spouse'])
     .assign(Relation="spouse")
     .rename(columns={'spouse':'Name2'})),
    
# father connections extracted from original name
    (targaryens[['name','Father']].dropna(subset=['Father']).assign(Relation="father")
     .rename(columns={'Father':'Name2'}))
], axis=0).drop("Relation",axis=1).values

T = nx.Graph()
T.add_nodes_from(nodes)
T.add_edges_from(edges)
T.nodes()

# extract nodes with no connections and remove them
no_connects = [i for i in nx.isolates(T)]
T.remove_nodes_from(no_connects)

# plot
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,12))
nx.draw_spring(T, alpha=.5, with_labels=True)

# 06-01

gene = pd.read_excel("gene_sequences.xlsx").set_index("Primary-Accession").dropna()
gene.head(1)

# 06-02
gene.corr()

# 06-03

# we also have DNA to amino-acid, replace U with T
D_to_A = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", 
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCA": "S",
    "TCC": "S", "TCG": "S", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCT": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y", "TAA": "-", "TAG": "-",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "-", "TGG": "W", "CGT": "R", "CGC": "R",
    "CGA": "R", "CGG": "R", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G"
}

def generate_cds(row):
    begin, dot, end = row.CDS_Range.split(".")
    # extract cds
    return row.mRNA_Sequence[int(begin)-1:int(end)]

def generate_aa(row):
    # generate triplets
    codons = [row.cds[i:i+3] for i in range(0, len(row.cds), 3)]
    # convert each codon to an amino acid and join the list together
    return "".join([D_to_A[codon] for codon in codons])

gene['cds'] = gene.apply(generate_cds, axis=1)
gene['aa_seq'] = gene.apply(generate_aa, axis=1)
# drop any that don't begin with M for aa-seq
gene = gene[gene.aa_seq.str.startswith("M")]

# 06-04

# calculate amino acid frequencies
aa_freqs = (gene.aa_seq.str.wrap(width=1)
    .str.split("\n",expand=True)
    .apply(lambda x: x.dropna().value_counts() / x.dropna().size, axis=1)
    .add_suffix("_f"))

n_exons = gene.Exons.str.split(";",expand=True).apply(lambda x: x.dropna().size, axis=1).to_frame("n_exons")

def extract_exon_length(value):
    start, dot, end = value.split(".")
    return int(end) - int(start)
mean_exon_length = gene.Exons.str.split(";",expand=True).stack().map(extract_exon_length).unstack().mean(axis=1).to_frame("mean_exon_length")

gc_content = ((gene.cds.str.count("G") + gene.cds.str.count("C")) / gene.cds.str.len()).to_frame("gc_content")

# calculate codon frequencies using the entire reference set
all_codons = gene.cds.str.wrap(width=3).str.split("\n",expand=True)
print(all_codons.shape)
# create our own reference set of frequencies here ready for CAI
freq_codons = all_codons.stack().value_counts().transform(lambda x: x/x.sum())

def calculate_cai_weight(codon, freq):
    # get amino acid
    aa = D_to_A[codon]
    # iterate over all codons, select all that share amino-acid
    syn_a = [j for j in D_to_A.keys() if D_to_A[j] == aa]
    syn_freq = [freq_codons[j] for j in syn_a]
    # print(codon, syn_a, syn_freq)
    w_i = freq / max(syn_freq)
    return w_i

# calculate w_i which is the weight of each codon as a ratio
# between observed frequency and the frequency of the most frequent synonymous codon
w_is = pd.Series([calculate_cai_weight(*codon_pkg) for i,codon_pkg in enumerate(freq_codons.items())],
                 index=freq_codons.index)

from numba import jit
import numpy as np

@jit
def calculate_cai(row):
    # to speed up calculation, drop none rows
    nrow = row.dropna()
    # logsum
    return (np.exp(nrow.replace(w_is.to_dict()).apply(np.log).sum()))**(1/nrow.size)

# now for each sequence, calculate CAI as the product (log sum) of all weights for each codon
cai = all_codons.apply(calculate_cai, axis=1).to_frame("cai")

new_geneset = pd.concat([
    gene, cai, gc_content, n_exons, mean_exon_length, aa_freqs
], axis=1)

# 06-05

gene.Description = gene.Description.str.extract("Homo sapiens (?P<Description>[a-zA-Z0-9\s\-\(\),/]+)",expand=True)

gene_abbrev = gene.Description.str.extract("[a-zA-Z0-9\s]+\((?P<abbrev>[a-zA-Z0-9]+)\)", expand=False).to_frame("Abbrev")

gene_transcript = gene.Description.str.extract(", transcript variant ([0-9]+)", expand=False).to_frame("Variant")

import numpy as np

def remove_shorts(e):
    if e == None:
        return e
    if type(e) == float:
        return None
    if len(e) < 4:
        return None
    else:
        return e

connecting_words = ['containing', 'subunit', 'factor', 'family', 'like', 'interacting', 'associated', 'with']
remove_connect = dict(zip(connecting_words, np.repeat(np.array([None]), len(connecting_words))))

words = (gene.Description.str.extract("(?P<textit>[a-zA-Z0-9\-\s/]+)", expand=False)
    .str.split(" ",expand=True)
    .applymap(remove_shorts)
    .replace(remove_connect))

reference_frequency = (words.stack().value_counts())
reference_frequency = reference_frequency.div(reference_frequency.sum())

word_frequencies = (words.replace(reference_frequency.to_dict()).fillna(1.))

# select least frequent index for each row.
idxs = word_frequencies.idxmin(axis=1)

selected = pd.Series([row[idxs[i]] for i,row in words.iterrows()], index=words.index).to_frame("Min_Word")

gene = pd.concat([
    gene, selected, gene_transcript, gene_abbrev
], axis=1)



