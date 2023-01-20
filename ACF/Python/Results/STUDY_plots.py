# -*- coding: utf-8 -*-
"""
Plots for study

@author: Yannis Schumann
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from scipy.stats import mode

palette = None #"Dark2" # for publication: None
# for dissertation, comment out for publication
#plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.Dark2.colors)



#------------------------------------------------------------------------------
#                   
#                               Including Covariable
#
#------------------------------------------------------------------------------
data = pd.read_csv("raw/petralia_covariable_comparison.csv", index_col=0)
dpt = data.pivot_table(index="Dataset", columns="Algorithm", values="Value")
best_acf_cov = dpt.idxmax(axis=1).loc["Petralia IRS"]
best_acf_no_cov = dpt.idxmax(axis=1).loc["Petralia IRS with Covariable"]
print(best_acf_cov, best_acf_no_cov)
data_no_cov = data[(data["Dataset"]=="Petralia IRS")&(data["Algorithm"]==best_acf_no_cov)]
data_cov = data[(data["Dataset"]=="Petralia IRS with Covariable")&(data["Algorithm"]==best_acf_cov)]

data_best = pd.concat((data_no_cov, data_cov), axis=0)

for rep_value in data_best["Repetition"].unique():
    rep_indices_no_cov = data_best[(data_best["Repetition"]==rep_value)&(data_best["Dataset"]=="Petralia IRS")].index.values
    rep_indices_cov = data_best[(data_best["Repetition"]==rep_value)&(data_best["Dataset"]=="Petralia IRS with Covariable")].index.values
    data_best.loc[rep_indices_cov, "Value"] = np.mean(data_best.loc[rep_indices_cov, "Value"].values)
    data_best.loc[rep_indices_cov, "Dataset"] = "Petralia et al\nWith Covariate"
    data_best.loc[rep_indices_no_cov, "Value"] = np.mean(data_best.loc[rep_indices_no_cov, "Value"].values)
    data_best.loc[rep_indices_no_cov, "Dataset"] = "Petralia et al\nCorrelation only"

fig, axs = plt.subplots(ncols=1, nrows=1)
sns.boxplot(data=data_best, x="Dataset", y="Value")
axs.set_ylabel("Macro-averaged $F_1$-Score")
axs.set_title("Consideration of Additional Covariates")
plt.savefig("petralia_covariable.pdf", bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
#                   
#                               ACF/KNN IMBALANCE
#
#------------------------------------------------------------------------------

imbalance_data = pd.read_csv("raw/ACFKNNImbalanceComparison.csv", index_col=0)
imbalance_data["imbalance"] = imbalance_data["a_size"]/(imbalance_data["a_size"]+imbalance_data["b_size"])


best_methods_acf_no_cov = imbalance_data[imbalance_data["Method"]=="ACF, correlation-only"].pivot_table(index="imbalance", columns="Algorithm", values="Value")
best_method_no_cov, _ = mode(np.argmax(best_methods_acf_no_cov.values, axis=1))
best_method_no_cov = best_methods_acf_no_cov.columns.values[best_method_no_cov[0]]
best_no_cov = imbalance_data[(imbalance_data["Method"]=="ACF, correlation-only")&(imbalance_data["Algorithm"]==best_method_no_cov)].copy()


if best_method_no_cov=="ACF+RF":
    best_no_cov["Algorithm"]="Best ACF (SVC/$\\bf{RF}$/Ridge)"
elif  best_method_no_cov=="ACF+SVC":
    best_no_cov["Algorithm"]="Best ACF ($\\bf{SVC}$/RF/Ridge)"
elif best_method_no_cov=="ACF+Ridge":
    best_no_cov["Algorithm"]="Best ACF (SVC/RF/$\\bf{Ridge}$)"
else:
    best_no_cov["Algorithm"]="Error" # won't occur

res = pd.concat((
    best_no_cov,
    imbalance_data[imbalance_data["Method"]=="KNN"],
    imbalance_data[imbalance_data["Method"]=="DBC"]
    ))
res = res.sort_values(by="Method")

sns.lineplot(data=res, x="imbalance", y="Value", hue="Algorithm", palette=palette, hue_order=["KNN", "KNN+ROs", "DBC", best_no_cov["Algorithm"].values[0]])
plt.xlabel(r"Class Imbalance $\frac{N_A}{N_A+N_B}$")
plt.ylabel("Macro-Averaged $F_1$-Score")
plt.title("Dependency of $F_1$-macro Score on Class Imbalance")
plt.savefig("simulation_classimbalance.pdf", bbox_inches = 'tight')
plt.show()

# baseline-correlation breaks symmetry for ACF --> at large number of A, the discriminative cross-correlation can be well estimated, but the correlation of B with A/C is less discriminative, which is why the classifier remains rather inaccurate even for larger numbers of B

#------------------------------------------------------------------------------
#                   
#                         SIZE OF TRAINING SET
#
#------------------------------------------------------------------------------
size_data = pd.read_csv("raw/ACFKNNSizeComparison.csv", index_col=0)

best_methods_acf_no_cov = size_data[size_data["Method"]=="ACF"].pivot_table(index="size", columns="Algorithm", values="Value")
best_method_no_cov, _ = mode(np.argmax(best_methods_acf_no_cov.values, axis=1))
best_method_no_cov = best_methods_acf_no_cov.columns.values[best_method_no_cov[0]]
best_no_cov = size_data[(size_data["Method"]=="ACF")&(size_data["Algorithm"]==best_method_no_cov)].copy()


if best_method_no_cov=="ACF+RF":
    best_no_cov["Algorithm"]="Best ACF (SVC/$\\bf{RF}$/Ridge)"
elif  best_method_no_cov=="ACF+SVC":
    best_no_cov["Algorithm"]="Best ACF ($\\bf{SVC}$/RF/Ridge)"
elif best_method_no_cov=="ACF+Ridge":
    best_no_cov["Algorithm"]="Best ACF (SVC/RF/$\\bf{Ridge}$)"
else:
    best_no_cov["Algorithm"]="Error" # won't occur
    

res = pd.concat((
    best_no_cov,
    size_data[size_data["Method"]=="KNN"],
    size_data[size_data["Method"]=="DBC"]
    ))


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
sns.lineplot(data=res, x="size", y="Value", hue="Algorithm", ax=axs[0], palette=palette, hue_order=["KNN", "KNN+ROs", "DBC", best_no_cov["Algorithm"].values[0]])
axs[0].set_ylabel(r"Macro-Averaged" "\n" "$F_1$-Score")
sns.lineplot(data=res[res["Algorithm"].isin(["KNN", "KNN+ROs"])], x="size", y="Neighbors", hue="Algorithm", ax=axs[1], palette=palette, hue_order=["KNN", "KNN+ROs", "DBC", best_no_cov["Algorithm"].values[0]])
axs[1].set_yscale("log")
axs[1].set_xlabel("Total Number of Instances")
axs[1].set_ylabel(r"Average No. of" "\n" "Nearest Neighbors")
fig.suptitle("Dependency of $F_1$-macro-Score on Total Number of Instances")

plt.savefig("simulation_datasetsize.pdf", bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
#                   
#                           TIME COMPLEXITY
#
#------------------------------------------------------------------------------

runtime_data = pd.read_csv("raw/FACF_KNN_TimeComplexity.csv", index_col=0)
runtime_data_untuned = runtime_data[runtime_data["Algorithm"]!="F-ACF, Optimized"]


fig, axs = plt.subplots(ncols=1, nrows=1)
sns.lineplot(data=runtime_data_untuned, x="Train Size", y="Runtime", style="Algorithm", hue="References", ax=axs)
axs.set_xlabel("Number of Training Instances")
axs.set_ylabel("Runtime per Predicted Test Instance in s")
axs.set_title("Runtime per Predicted Test Instance for KNN, DBC and F-ACF")

plt.savefig("simulation_runtime.pdf", bbox_inches = 'tight')
plt.show()

# suppl. information
a = runtime_data.pivot_table(index="Train Size", columns=["Algorithm", "References"], values="Value")
#
tot_untuned = np.concatenate(((a["F-ACF"]["10"]>a["F-DBC"]["10"]).values, (a["F-ACF"]["20"]>a["F-DBC"]["20"]).values, (a["F-ACF"]["30"]>a["F-DBC"]["30"]).values))
print("Average F-ACF is in {} percent of the cases better than the respective F-DBC counterpart.".format(round(np.sum(tot_untuned)/len(tot_untuned)*100, 2)))
tot_tuned = np.concatenate(((a["F-ACF, Optimized"]["10"]>a["F-DBC"]["10"]).values, (a["F-ACF, Optimized"]["20"]>a["F-DBC"]["20"]).values, (a["F-ACF, Optimized"]["30"]>a["F-DBC"]["30"]).values))
print("Average F-ACF is in {} percent of the cases better than the respective F-DBC counterpart.".format(round(np.sum(tot_tuned)/len(tot_tuned)*100, 2)))

fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True)

for index, alg in enumerate(["F-ACF", "F-ACF, Optimized"]):
    references_list = []
    algs_list = []
    win_list = []
    for n_ref in ["10","20","30"]:
        print(n_ref)
        f_dbc_data = runtime_data[(runtime_data["Algorithm"]=="F-DBC")&(runtime_data["References"]==n_ref)]["Value"].values
        f_acf_data = runtime_data[(runtime_data["Algorithm"] == alg) & (runtime_data["References"] == n_ref)]["Value"].values
        references_list.extend([n_ref, n_ref])
        algs_list.extend([alg, "F-DBC"])
        win_list.extend([np.sum(f_acf_data>f_dbc_data), np.sum(f_acf_data<f_dbc_data)])
    comp = pd.DataFrame({"References":references_list, "Algorithm":algs_list, "Wins": win_list})
    sns.barplot(data=comp, x="References", y="Wins", hue="Algorithm", ax=axs[index])
axs[0].set_title("No Hyperparameter Optimization")
axs[1].set_title("With Hyperparameter Optimization")
fig.suptitle("Comparison of F-ACF and F-DBC")
fig.tight_layout()
plt.savefig("suppl_facf_fdbc.pdf", bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
#                   
#                           NOISE STABILITY
#
#------------------------------------------------------------------------------

noise_data = pd.read_csv("raw/ACFKNNNoiseComparison.csv", index_col=0)

# generate column with relative noise
noise_data["relative_noise"] = noise_data["sigma"]/noise_data["diff_mu_12_mu_11"]

for mv in noise_data["mv"].unique():
    noise_data.loc[noise_data["mv"]==mv, "relative_noise"] = np.mean(noise_data.loc[noise_data["mv"]==mv, "relative_noise"].values)

best_methods_acf_no_cov = noise_data[noise_data["Method"]=="ACF, correlation-only"].pivot_table(index="relative_noise", columns="Algorithm", values="Value")
best_method_no_cov, _ = mode(np.argmax(best_methods_acf_no_cov.values, axis=1))
best_method_no_cov = best_methods_acf_no_cov.columns.values[best_method_no_cov[0]]
best_no_cov = noise_data[(noise_data["Method"]=="ACF, correlation-only")&(noise_data["Algorithm"]==best_method_no_cov)].copy()
if best_method_no_cov=="ACF+RF":
    best_no_cov["Algorithm"]="Best ACF (SVC/$\\bf{RF}$/Ridge)"
elif  best_method_no_cov=="ACF+SVC":
    best_no_cov["Algorithm"]="Best ACF ($\\bf{SVC}$/RF/Ridge)"
elif best_method_no_cov=="ACF+Ridge":
    best_no_cov["Algorithm"]="Best ACF (SVC/RF/$\\bf{Ridge}$)"
else:
    best_no_cov["Algorithm"]="Error" # won't occur


best_methods_acf_cov = noise_data[noise_data["Method"]=="ACF, with covariable"].pivot_table(index="relative_noise", columns="Algorithm", values="Value")
best_method_cov, _ = mode(np.argmax(best_methods_acf_no_cov.values, axis=1))
best_method_cov = best_methods_acf_cov.columns.values[best_method_cov[0]]
best_cov = noise_data[(noise_data["Method"]=="ACF, with covariable")&(noise_data["Algorithm"]==best_method_cov)].copy()

if best_method_no_cov=="ACF+RF":
    best_cov["Algorithm"]="Best ACF (SVC/$\\bf{RF}$/Ridge)\n With Covariate"
elif  best_method_no_cov=="ACF+SVC":
    best_cov["Algorithm"]="Best ACF ($\\bf{SVC}$/RF/Ridge)\n With Covariate"
elif best_method_no_cov=="ACF+Ridge":
    best_cov["Algorithm"]="Best ACF (SVC/RF/$\\bf{Ridge}$)\n With Covariate"
else:
    best_cov["Algorithm"]="Error" # won't occur


best_cov_method = noise_data.loc[noise_data["Method"]=="Covariable Alone"].pivot_table(index="Algorithm", values="Value").idxmax().values[0]
best_cov_value = noise_data.loc[(noise_data["Method"]=="Covariable Alone")&(noise_data["Algorithm"]==best_cov_method), "Value"].mean()

noise_data = pd.concat((
    best_no_cov,
    best_cov,
    noise_data[noise_data["Method"]=="KNN"],
    noise_data[noise_data["Method"]=="DBC"]
    ))

fig, axs = plt.subplots(nrows=1, ncols=1)
sns.lineplot(data=noise_data, x="relative_noise", y="Value", hue="Algorithm", palette=palette, ax=axs, hue_order=["KNN", "KNN+ROs", "DBC", best_no_cov["Algorithm"].values[0], best_cov["Algorithm"].values[0]])
axs.axhline(y=best_cov_value, color="black", linestyle="--")
axs.text(0.5,best_cov_value, "Covariate Only")
plt.xlabel(r"Average Relative Noise $\frac{\sigma}{\mu_{AA}-\mu_{AB}}$")
plt.ylabel(r"Macro-Averaged $F_1$-Score")
plt.title("Dependency of $F_1$-macro-Score on Relative Noise")

plt.savefig("simulation_noise.pdf", bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
#                   
#                               ACF/F-ACF
#
#------------------------------------------------------------------------------

acf_facf_data = pd.read_csv("raw/ACFFACFComparison.csv", index_col=0)

# generate column with relative noise
acf_facf_data["relative_noise"] = acf_facf_data["sigma"]/acf_facf_data["diff_mu_12_mu_11"]

# average relative noise
for mv in acf_facf_data["mv"].unique():
    mv_indices = acf_facf_data[acf_facf_data["mv"]==mv].index.values
    acf_facf_data.loc[mv_indices, "relative_noise"] = np.mean(acf_facf_data.loc[mv_indices, "relative_noise"].values)

# lists for results

x_acf = [] # num. of points
y_acf = [] # relative noise
z_acf = [] # f1-score ACF

z_facf = [] # f1-score F-ACF

for rel_noise in acf_facf_data["relative_noise"].values:
    for point in  acf_facf_data["points"].unique():
        # skip ACF result here
        if np.isnan(point):
            continue
        # select f_acf for this relative noise and this number of points
        f_acf_indices = acf_facf_data[(acf_facf_data["relative_noise"]==rel_noise)&(acf_facf_data["points"]==point)].index.values # not necessary to select F-ACF, since ACF will have nan points
        # select corresponding ACF runs
        acf_indices = acf_facf_data[(acf_facf_data["relative_noise"]==rel_noise)&(acf_facf_data["points"].isna())].index.values # not necessary to select F-ACF, since ACF will have nan points
        
        x_acf.append(point)
        y_acf.append(rel_noise)
        z_acf.append(np.mean(acf_facf_data.loc[acf_indices, "score"]))
        z_facf.append(np.mean(acf_facf_data.loc[f_acf_indices, "score"]))


fig = plt.figure()
ax = Axes3D(fig)
sfacf = ax.plot_trisurf(x_acf, y_acf, z_facf, label="F-ACF (SVC)")
sfacf._facecolors2d=sfacf._facecolor3d
sfacf._edgecolors2d=sfacf._edgecolor3d
sacf = ax.plot_trisurf(x_acf, y_acf, z_acf, label="ACF (SVC)")
sacf._facecolors2d=sacf._facecolor3d
sacf._edgecolors2d=sacf._edgecolor3d
ax.set_ylabel(r"Average Relative Noise $\frac{\sigma}{\mu_{AA}-\mu_{AB}}$")
ax.set_xlabel(r"F-ACF References per Class")
ax.set_zlabel(r"Macro-Averaged $F_1$-Score")
ax.legend(loc="lower left")
fig.suptitle(r"Trade-off between Computational""\n""Performance and Macro-Averaged $F_1$-Score")
ax.view_init(elev=10, azim=112)
plt.savefig("simulation_fasterprediction.pdf", bbox_inches = 'tight')

plt.show()


#------------------------------------------------------------------------------
#                   
#                               scRNA-seq
#
#------------------------------------------------------------------------------

scRNA_data = pd.read_csv("raw/full-comp-scRNAseq.csv", index_col=0)
# summary
#
#scRNA_data.pivot_table(index="Algorithm", columns="Dataset", values="Value", aggfunc=lambda x: str(round(np.mean(x),3))+"+-"+str(round(np.std(x),3)))

# define hatchings for plot
#hatches = ["++","..","\\\\", "//"]
# create plots
fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True)

# select and plot PBMC data / 10X Genomics
data_pbmc = scRNA_data[scRNA_data["Dataset"]=="3kPBMC"]
piv_pbmc = data_pbmc.pivot_table(index="Algorithm", values="Value")
best_acf_pbmc = piv_pbmc.loc[["ACF+RF","ACF+Ridge","ACF+SVC"]].idxmax().values[0]
best_listwise_pbmc = piv_pbmc.loc[["RF","Ridge","SVC"]].idxmax().values[0]

data_pbmc = data_pbmc[data_pbmc["Algorithm"].isin(["KNN", "DBC", best_acf_pbmc, best_listwise_pbmc])]
if best_acf_pbmc=="ACF+RF":
    data_pbmc.loc[data_pbmc["Algorithm"]==best_acf_pbmc, "Algorithm"] = "Best ACF\n(SVC/$\\bf{RF}$/Ridge)"
elif best_acf_pbmc=="ACF+Ridge":
    data_pbmc.loc[data_pbmc["Algorithm"]==best_acf_pbmc, "Algorithm"] = "Best ACF\n(SVC/RF/$\\bf{Ridge}$)"
elif best_acf_pbmc=="ACF+SVC":
    data_pbmc.loc[data_pbmc["Algorithm"]==best_acf_pbmc, "Algorithm"] = "Best ACF\n($\\bf{SVC}$/RF/Ridge)"

if best_listwise_pbmc=="RF":
    data_pbmc.loc[data_pbmc["Algorithm"]==best_listwise_pbmc, "Algorithm"] = "Best Listwise Deletion\n(SVC/$\\bf{RF}$/Ridge)"
elif best_listwise_pbmc=="Ridge":
    data_pbmc.loc[data_pbmc["Algorithm"]==best_listwise_pbmc, "Algorithm"] = "Best Listwise Deletion\n(SVC/RF/$\\bf{Ridge}$)"
elif best_listwise_pbmc=="SVC":
    data_pbmc.loc[data_pbmc["Algorithm"]==best_listwise_pbmc, "Algorithm"] = "Best Listwise Deletion\n($\\bf{SVC}$/RF/Ridge)"


bar = sns.barplot(data=data_pbmc, x="Algorithm", y="Value", ax=axs[0], palette=palette)
# for i,thisbar in enumerate(bar.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i])
axs[0].set_ylabel(r"Macro-Averaged $F_1$-Score")
axs[0].set_title("10X Genomics")
axs[0].tick_params(axis='x', rotation=90)
axs[0].set_xlabel(None)


# select and plot PBMC data / 10X Genomics
data_xin = scRNA_data[scRNA_data["Dataset"]=="Xin"]
piv_xin = data_xin.pivot_table(index="Algorithm", values="Value")
best_acf_xin = piv_xin.loc[["ACF+RF","ACF+Ridge","ACF+SVC"]].idxmax().values[0]
best_listwise_xin = piv_xin.loc[["RF","Ridge","SVC"]].idxmax().values[0]

data_xin = data_xin[data_xin["Algorithm"].isin(["KNN", "DBC", best_acf_xin, best_listwise_xin])]


if best_acf_xin=="ACF+RF":
    data_xin.loc[data_xin["Algorithm"]==best_acf_xin, "Algorithm"] = "Best ACF\n(SVC/$\\bf{RF}$/Ridge)"
elif best_acf_xin=="ACF+Ridge":
    data_xin.loc[data_xin["Algorithm"]==best_acf_xin, "Algorithm"] = "Best ACF\n(SVC/RF/$\\bf{Ridge}$)"
elif best_acf_xin=="ACF+SVC":
    data_xin.loc[data_xin["Algorithm"]==best_acf_xin, "Algorithm"] = "Best ACF\n($\\bf{SVC}$/RF/Ridge)"

if best_listwise_pbmc=="RF":
    data_xin.loc[data_xin["Algorithm"]==best_listwise_xin, "Algorithm"] = "Best Listwise Deletion\n(SVC/$\\bf{RF}$/Ridge)"
elif best_listwise_pbmc=="Ridge":
    data_xin.loc[data_xin["Algorithm"]==best_listwise_xin, "Algorithm"] = "Best Listwise Deletion\n(SVC/RF/$\\bf{Ridge}$)"
elif best_listwise_pbmc=="SVC":
    data_xin.loc[data_xin["Algorithm"]==best_listwise_xin, "Algorithm"] = "Best Listwise Deletion\n($\\bf{SVC}$/RF/Ridge)"
    

bar = sns.barplot(data=data_xin, x="Algorithm", y="Value", ax=axs[1], palette=palette)
# for i,thisbar in enumerate(bar.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i])
axs[1].set_title("Xin et al")
axs[1].tick_params(axis='x', rotation=90)
axs[1].set_xlabel(None)
axs[1].set_ylabel(None)



# select and plot PBMC data / 10X Genomics
data_baron = scRNA_data[scRNA_data["Dataset"]=="Baron"]
piv_baron = data_baron.pivot_table(index="Algorithm", values="Value")
best_acf_baron = piv_baron.loc[["ACF+RF","ACF+Ridge","ACF+SVC"]].idxmax().values[0]
best_listwise_baron = piv_baron.loc[["RF","Ridge","SVC"]].idxmax().values[0]

data_baron = data_baron[data_baron["Algorithm"].isin(["KNN", "DBC", best_acf_baron, best_listwise_baron])]


if best_acf_baron=="ACF+RF":
    data_baron.loc[data_baron["Algorithm"]==best_acf_baron, "Algorithm"] = "Best ACF\n(SVC/$\\bf{RF}$/Ridge)"
elif best_acf_baron=="ACF+Ridge":
    data_baron.loc[data_baron["Algorithm"]==best_acf_baron, "Algorithm"] = "Best ACF\n(SVC/RF/$\\bf{Ridge}$)"
elif best_acf_baron=="ACF+SVC":
    data_baron.loc[data_baron["Algorithm"]==best_acf_baron, "Algorithm"] = "Best ACF\n($\\bf{SVC}$/RF/Ridge)"

if best_listwise_baron=="RF":
    data_baron.loc[data_baron["Algorithm"]==best_listwise_baron, "Algorithm"] = "Best Listwise Deletion\n(SVC/$\\bf{RF}$/Ridge)"
elif best_listwise_baron=="Ridge":
    data_baron.loc[data_baron["Algorithm"]==best_listwise_baron, "Algorithm"] = "Best Listwise Deletion\n(SVC/RF/$\\bf{Ridge}$)"
elif best_listwise_baron=="SVC":
    data_baron.loc[data_baron["Algorithm"]==best_listwise_baron, "Algorithm"] = "Best Listwise Deletion\n($\\bf{SVC}$/RF/Ridge)"
    

data_baron.loc[data_baron["Algorithm"]==best_acf_baron, "Algorithm"] = "Best ACF\n(SVC/RF/Ridge)"
data_baron.loc[data_baron["Algorithm"]==best_listwise_baron, "Algorithm"] = "Best Listwise Deletion\n(SVC/RF/Ridge)"

bar = sns.barplot(data=data_baron, x="Algorithm", y="Value", ax=axs[2], palette=palette)
# for i,thisbar in enumerate(bar.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i])
axs[2].set_title("Baron et al")
axs[2].tick_params(axis='x', rotation=90)
axs[2].set_xlabel(None)
axs[2].set_ylabel(None)
fig.suptitle("Comparison on Three scRNA-seq Datasets")
fig.tight_layout()
plt.savefig("scrnaseq-comp.pdf", bbox_inches = 'tight')
plt.show()

#------------------------------------------------------------------------------
#                   
#                               PROTEOMICS
#
#------------------------------------------------------------------------------
proteomics_data = pd.read_csv("raw/full-comp-proteomics.csv", index_col=0)

# summary
# proteomics_data.pivot_table(index="Algorithm", columns="Dataset", values="Value", aggfunc=lambda x: str(round(np.mean(x),3))+"+-"+str(round(np.std(x),3)))
# define hatchings for plot
#hatches = ["++","..","\\\\", "//"]
# create plots
fig, axs = plt.subplots(nrows=1, ncols=4, sharey=True)

# select and plot petralia, corrected
data_petralia_IRS = proteomics_data[proteomics_data["Dataset"]=="Petralia(IRS)"]

piv_petralia_irs = data_petralia_IRS.pivot_table(index="Algorithm", values="Value")
best_acf_petralia_irs = piv_petralia_irs.loc[["ACF+RF","ACF+Ridge","ACF+SVC"]].idxmax().values[0]
best_listwise_petralia_irs = piv_petralia_irs.loc[["RF","Ridge","SVC"]].idxmax().values[0]

data_petralia_IRS = data_petralia_IRS[data_petralia_IRS["Algorithm"].isin(["KNN", "DBC", best_acf_petralia_irs, best_listwise_petralia_irs])]

if best_acf_petralia_irs=="ACF+RF":
    data_petralia_IRS.loc[data_petralia_IRS["Algorithm"]==best_acf_petralia_irs, "Algorithm"] = "Best ACF\n(SVC/$\\bf{RF}$/Ridge)"
elif best_acf_petralia_irs=="ACF+Ridge":
    data_petralia_IRS.loc[data_petralia_IRS["Algorithm"]==best_acf_petralia_irs, "Algorithm"] = "Best ACF\n(SVC/RF/$\\bf{Ridge}$)"
elif best_acf_petralia_irs=="ACF+SVC":
    data_petralia_IRS.loc[data_petralia_IRS["Algorithm"]==best_acf_petralia_irs, "Algorithm"] = "Best ACF\n($\\bf{SVC}$/RF/Ridge)"

if best_listwise_petralia_irs=="RF":
    data_petralia_IRS.loc[data_petralia_IRS["Algorithm"]==best_listwise_petralia_irs, "Algorithm"] = "Best Listwise Deletion\n(SVC/$\\bf{RF}$/Ridge)"
elif best_listwise_petralia_irs=="Ridge":
    data_petralia_IRS.loc[data_petralia_IRS["Algorithm"]==best_listwise_petralia_irs, "Algorithm"] = "Best Listwise Deletion\n(SVC/RF/$\\bf{Ridge}$)"
elif best_listwise_petralia_irs=="SVC":
    data_petralia_IRS.loc[data_petralia_IRS["Algorithm"]==best_listwise_petralia_irs, "Algorithm"] = "Best Listwise Deletion\n($\\bf{SVC}$/RF/Ridge)"


bar = sns.barplot(data=data_petralia_IRS, x="Algorithm", y="Value", ax=axs[0], palette=palette)
# for i,thisbar in enumerate(bar.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i])
axs[0].set_ylabel(r"Macro-Averaged $F_1$-Score")
axs[0].set_title(r"Petralia et al" "\n" "Corrected")
axs[0].tick_params(axis='x', rotation=90)
axs[0].set_xlabel(None)

# select and plot petralia, raw
data_petralia_raw = proteomics_data[proteomics_data["Dataset"]=="Petralia(raw)"]

# select and plot petralia, corrected
data_petralia_raw = proteomics_data[proteomics_data["Dataset"]=="Petralia(raw)"]

piv_petralia_raw = data_petralia_raw.pivot_table(index="Algorithm", values="Value")
best_acf_petralia_raw = piv_petralia_raw.loc[["ACF+RF","ACF+Ridge","ACF+SVC"]].idxmax().values[0]
best_listwise_petralia_raw = piv_petralia_raw.loc[["RF","Ridge","SVC"]].idxmax().values[0]

data_petralia_raw = data_petralia_raw[data_petralia_raw["Algorithm"].isin(["KNN", "DBC", best_acf_petralia_raw, best_listwise_petralia_raw])]

if best_acf_petralia_raw=="ACF+RF":
    data_petralia_raw.loc[data_petralia_raw["Algorithm"]==best_acf_petralia_raw, "Algorithm"] = "Best ACF\n(SVC/$\\bf{RF}$/Ridge)"
elif best_acf_petralia_raw=="ACF+Ridge":
    data_petralia_raw.loc[data_petralia_raw["Algorithm"]==best_acf_petralia_raw, "Algorithm"] = "Best ACF\n(SVC/RF/$\\bf{Ridge}$)"
elif best_acf_petralia_raw=="ACF+SVC":
    data_petralia_raw.loc[data_petralia_raw["Algorithm"]==best_acf_petralia_raw, "Algorithm"] = "Best ACF\n($\\bf{SVC}$/RF/Ridge)"

if best_listwise_petralia_raw=="RF":
    data_petralia_raw.loc[data_petralia_raw["Algorithm"]==best_listwise_petralia_raw, "Algorithm"] = "Best Listwise Deletion\n(SVC/$\\bf{RF}$/Ridge)"
elif best_listwise_petralia_raw=="Ridge":
    data_petralia_raw.loc[data_petralia_raw["Algorithm"]==best_listwise_petralia_raw, "Algorithm"] = "Best Listwise Deletion\n(SVC/RF/$\\bf{Ridge}$)"
elif best_listwise_petralia_raw=="SVC":
    data_petralia_raw.loc[data_petralia_raw["Algorithm"]==best_listwise_petralia_raw, "Algorithm"] = "Best Listwise Deletion\n($\\bf{SVC}$/RF/Ridge)"


bar = sns.barplot(data=data_petralia_raw, x="Algorithm", y="Value", ax=axs[1], palette=palette)
# for i,thisbar in enumerate(bar.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i])
axs[1].set_ylabel(None)
axs[1].set_title(r"Petralia et al" "\n" "Raw")
axs[1].tick_params(axis='x', rotation=90)
axs[1].set_xlabel(None)


# select and plot krug et al, corrected
data_brca_IRS = proteomics_data[proteomics_data["Dataset"]=="BRCA(IRS)"]

piv_brca_irs = data_brca_IRS.pivot_table(index="Algorithm", values="Value")
best_acf_brca_irs = piv_brca_irs.loc[["ACF+RF","ACF+Ridge","ACF+SVC"]].idxmax().values[0]
best_listwise_brca_irs = piv_brca_irs.loc[["RF","Ridge","SVC"]].idxmax().values[0]

data_brca_IRS = data_brca_IRS[data_brca_IRS["Algorithm"].isin(["KNN", "DBC", best_acf_brca_irs, best_listwise_brca_irs])]

if best_acf_brca_irs=="ACF+RF":
    data_brca_IRS.loc[data_brca_IRS["Algorithm"]==best_acf_brca_irs, "Algorithm"] = "Best ACF\n(SVC/$\\bf{RF}$/Ridge)"
elif best_acf_brca_irs=="ACF+Ridge":
    data_brca_IRS.loc[data_brca_IRS["Algorithm"]==best_acf_brca_irs, "Algorithm"] = "Best ACF\n(SVC/RF/$\\bf{Ridge}$)"
elif best_acf_brca_irs=="ACF+SVC":
    data_brca_IRS.loc[data_brca_IRS["Algorithm"]==best_acf_brca_irs, "Algorithm"] = "Best ACF\n($\\bf{SVC}$/RF/Ridge)"

if best_listwise_brca_irs=="RF":
    data_brca_IRS.loc[data_brca_IRS["Algorithm"]==best_listwise_brca_irs, "Algorithm"] = "Best Listwise Deletion\n(SVC/$\\bf{RF}$/Ridge)"
elif best_listwise_brca_irs=="Ridge":
    data_brca_IRS.loc[data_brca_IRS["Algorithm"]==best_listwise_brca_irs, "Algorithm"] = "Best Listwise Deletion\n(SVC/RF/$\\bf{Ridge}$)"
elif best_listwise_brca_irs=="SVC":
    data_brca_IRS.loc[data_brca_IRS["Algorithm"]==best_listwise_brca_irs, "Algorithm"] = "Best Listwise Deletion\n($\\bf{SVC}$/RF/Ridge)"


bar = sns.barplot(data=data_brca_IRS, x="Algorithm", y="Value", ax=axs[2], palette=palette)
# for i,thisbar in enumerate(bar.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i])
axs[2].set_ylabel(None)
axs[2].set_title(r"Krug et al" "\n" "Corrected")
axs[2].tick_params(axis='x', rotation=90)
axs[2].set_xlabel(None)


# select and plot petralia, raw
data_brca_raw = proteomics_data[proteomics_data["Dataset"]=="BRCA(raw)"]

piv_brca_raw = data_brca_raw.pivot_table(index="Algorithm", values="Value")
best_acf_brca_raw = piv_brca_raw.loc[["ACF+RF","ACF+Ridge","ACF+SVC"]].idxmax().values[0]
best_listwise_brca_raw = piv_brca_raw.loc[["RF","Ridge","SVC"]].idxmax().values[0]

data_brca_raw = data_brca_raw[data_brca_raw["Algorithm"].isin(["KNN", "DBC", best_acf_brca_raw, best_listwise_brca_raw])]

if best_acf_brca_raw=="ACF+RF":
    data_brca_raw.loc[data_brca_raw["Algorithm"]==best_acf_brca_raw, "Algorithm"] = "Best ACF\n(SVC/$\\bf{RF}$/Ridge)"
elif best_acf_brca_raw=="ACF+Ridge":
    data_brca_raw.loc[data_brca_raw["Algorithm"]==best_acf_brca_raw, "Algorithm"] = "Best ACF\n(SVC/RF/$\\bf{Ridge}$)"
elif best_acf_brca_raw=="ACF+SVC":
    data_brca_raw.loc[data_brca_raw["Algorithm"]==best_acf_brca_raw, "Algorithm"] = "Best ACF\n($\\bf{SVC}$/RF/Ridge)"

if best_listwise_brca_raw=="RF":
    data_brca_raw.loc[data_brca_raw["Algorithm"]==best_listwise_brca_raw, "Algorithm"] = "Best Listwise Deletion\n(SVC/$\\bf{RF}$/Ridge)"
elif best_listwise_brca_raw=="Ridge":
    data_brca_raw.loc[data_brca_raw["Algorithm"]==best_listwise_brca_raw, "Algorithm"] = "Best Listwise Deletion\n(SVC/RF/$\\bf{Ridge}$)"
elif best_listwise_brca_raw=="SVC":
    data_brca_raw.loc[data_brca_raw["Algorithm"]==best_listwise_brca_raw, "Algorithm"] = "Best Listwise Deletion\n($\\bf{SVC}$/RF/Ridge)"


bar = sns.barplot(data=data_brca_raw, x="Algorithm", y="Value", ax=axs[3], palette=palette)
# for i,thisbar in enumerate(bar.patches):
#     # Set a different hatch for each bar
#     thisbar.set_hatch(hatches[i])
axs[3].set_ylabel(None)
axs[3].set_title(r"Krug et al" "\n" "Raw")
axs[3].tick_params(axis='x', rotation=90)
axs[3].set_xlabel(None)


fig.suptitle("Comparison on Two Proteomic Datasets")
fig.tight_layout()
plt.savefig("proteomic-comp.pdf", bbox_inches = 'tight')

plt.show()


###############################################################################
#
#                        Cross-Correlations
#
###############################################################################

def plot_variable_importance(df, filename, n_highlight=4, title=None):
    
    df = df.pivot_table(index="Class", columns="Left-Out Correlation", values="F1 Score", aggfunc=lambda x: round(np.mean(x),2))
    fig, axs = plt.subplots(nrows=1, ncols=1)
    sns.heatmap(df, annot=True, ax=axs, fmt=".2g", cbar_kws={'label': 'Importance'})
    # highlight the n highest off-diagonal elements
    df2 = df.copy()
    np.fill_diagonal(df2.values, -np.inf)
    for _ in range(n_highlight):
        ax0_idx, ax1_idx = np.unravel_index(df2.values.argmax(), df2.shape)
        df2.values[ax0_idx, ax1_idx] = -np.inf
        axs.add_patch(Rectangle((ax1_idx, ax0_idx), 1, 1, fill=False, edgecolor='blue', lw=3))
    axs.set_ylabel("Predicted Class")
    axs.set_xlabel("Correlation to Class")
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename+"_heatmap.pdf", bbox_inches = 'tight')
    plt.show()
    

xin_importance = pd.read_csv("raw/xin_acf_feature_importance.csv", index_col=0)
petralia_importance = pd.read_csv("raw/petralia_acf_feature_importance.csv", index_col=0)

plot_variable_importance(xin_importance, "xinvarimp", title="Importance of Correlations" "\n" "Xin et al")
plot_variable_importance(petralia_importance, "petraliavarimp", title="Importance of Correlations" "\n" "Petralia et al")