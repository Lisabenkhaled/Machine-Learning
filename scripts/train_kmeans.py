#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from train.io import TARGET_COL, TIME_COL
from train.metrics import weighted_accuracy


@dataclass
class Preprocessor:
    featurecols: list[str]
    medians: pd.Series
    means: pd.Series
    stds: pd.Series


@dataclass
class KMeansClusterModel:
    preprocessor: Preprocessor
    kmeans: KMeans
    clustertargetmean: dict[int, float]
    globaltargetmean: float
    selectedk: int
    init: str
    ninit: int
    randomstate: int


def selectfeaturecolumns(df: pd.DataFrame) -> list[str]:
    numericcols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numericcols if c != TARGET_COL]


def fitpreprocessor(df: pd.DataFrame, featurecols: list[str]) -> Preprocessor:
    X = df[featurecols].replace([np.inf, -np.inf], np.nan)
    medians = X.median(numeric_only=True).fillna(0.0)
    Ximp = X.fillna(medians)
    means = Ximp.mean(numeric_only=True)
    stds = Ximp.std(ddof=0, numeric_only=True).replace(0.0, 1.0).fillna(1.0)
    return Preprocessor(featurecols=featurecols, medians=medians, means=means, stds=stds)


def transformfeatures(df: pd.DataFrame, prep: Preprocessor) -> np.ndarray:
    X = df[prep.featurecols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(prep.medians)
    X = (X - prep.means) / prep.stds
    return X.to_numpy(dtype=float)


def buildclustertargetmapping(labels: np.ndarray, y: np.ndarray) -> tuple[dict[int, float], float]:
    globalmean = float(np.mean(y))
    mapping: dict[int, float] = {}
    for c in np.unique(labels):
        mask = labels == c
        mapping[int(c)] = float(np.mean(y[mask])) if np.any(mask) else globalmean
    return mapping, globalmean


def applyclustermapping(labels: np.ndarray, mapping: dict[int, float], fallback: float) -> np.ndarray:
    return np.array([mapping.get(int(lbl), fallback) for lbl in labels], dtype=float)


def fitkmeansmodel(traindf: pd.DataFrame, k: int, init: str, ninit: int, randomstate: int) -> KMeansClusterModel:
    featurecols = selectfeaturecolumns(traindf)
    prep = fitpreprocessor(traindf, featurecols)
    Xtrain = transformfeatures(traindf, prep)
    ytrain = traindf[TARGET_COL].to_numpy(dtype=float)
    kmeans = KMeans(n_clusters=k, init=init, n_init=ninit, random_state=randomstate)
    labels = kmeans.fit_predict(Xtrain)
    mapping, globalmean = buildclustertargetmapping(labels, ytrain)
    return KMeansClusterModel(
        preprocessor=prep,
        kmeans=kmeans,
        clustertargetmean=mapping,
        globaltargetmean=globalmean,
        selectedk=k,
        init=init,
        ninit=ninit,
        randomstate=randomstate,
    )


def predictwithmodel(df: pd.DataFrame, model: KMeansClusterModel) -> tuple[pd.DataFrame, np.ndarray]:
    X = transformfeatures(df, model.preprocessor)
    labels = model.kmeans.predict(X)
    ypred = applyclustermapping(labels, model.clustertargetmean, model.globaltargetmean)
    out = pd.DataFrame({TIME_COL: df[TIME_COL].copy(), TARGET_COL: ypred, "cluster": labels})
    return out, labels


def regressionmetrics(ytrue: np.ndarray, ypred: np.ndarray) -> tuple[float, float, float]:
    wa = float(weighted_accuracy(ytrue, ypred))
    rmse = float(np.sqrt(np.mean((ytrue - ypred) ** 2)))
    mae = float(np.mean(np.abs(ytrue - ypred)))
    return wa, rmse, mae


def expand_n_init_grid(values: Iterable[int] | None, fallback: int) -> list[int]:
    vals = list(values) if values is not None else []
    return vals if vals else [fallback]


def expand_init_grid(values: Iterable[str] | None, fallback: str) -> list[str]:
    vals = list(values) if values is not None else []
    return vals if vals else [fallback]


def timeseriescvsearch(
    traindf: pd.DataFrame,
    kvalues: list[int],
    nsplits: int,
    initgrid: list[str],
    ninitgrid: list[int],
    randomstate: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    ordered = traindf.sort_values(TIME_COL).reset_index(drop=True).copy()
    tscv = TimeSeriesSplit(n_splits=nsplits)
    rows: list[dict] = []

    for k in kvalues:
        for init in initgrid:
            for ninit in ninitgrid:
                for foldid, (tridx, vaidx) in enumerate(tscv.split(ordered), start=1):
                    foldtrain = ordered.iloc[tridx].copy()
                    foldvalid = ordered.iloc[vaidx].copy()
                    model = fitkmeansmodel(foldtrain, k=k, init=init, ninit=ninit, randomstate=randomstate)
                    predtrain, _ = predictwithmodel(foldtrain, model)
                    predvalid, _ = predictwithmodel(foldvalid, model)
                    ytrain = foldtrain[TARGET_COL].to_numpy(dtype=float)
                    yvalid = foldvalid[TARGET_COL].to_numpy(dtype=float)
                    yhattrain = predtrain[TARGET_COL].to_numpy(dtype=float)
                    yhatvalid = predvalid[TARGET_COL].to_numpy(dtype=float)
                    watrain, rmsetrain, maetrain = regressionmetrics(ytrain, yhattrain)
                    wavalid, rmsevalid, maevalid = regressionmetrics(yvalid, yhatvalid)
                    rows.append({
                        "k": k,
                        "init": init,
                        "ninit": ninit,
                        "fold": foldid,
                        "trainsize": len(foldtrain),
                        "validsize": len(foldvalid),
                        "trainwa": watrain,
                        "trainrmse": rmsetrain,
                        "trainmae": maetrain,
                        "validwa": wavalid,
                        "validrmse": rmsevalid,
                        "validmae": maevalid,
                        "wagaptrainminusvalid": watrain - wavalid,
                        "rmsegapvalidminustrain": rmsevalid - rmsetrain,
                        "inertiatrain": float(model.kmeans.inertia_),
                        "randomstate": randomstate,
                    })

    cvdetail = pd.DataFrame(rows)
    cvsummary = (
        cvdetail.groupby(["k", "init", "ninit"], as_index=False)
        .agg(
            cvvalidwamean=("validwa", "mean"),
            cvvalidwastd=("validwa", "std"),
            cvvalidrmsemean=("validrmse", "mean"),
            cvvalidrmsestd=("validrmse", "std"),
            cvvalidmaemean=("validmae", "mean"),
            cvtrainwamean=("trainwa", "mean"),
            cvtrainrmsemean=("trainrmse", "mean"),
            cvwagapmean=("wagaptrainminusvalid", "mean"),
            cvrmsegapmean=("rmsegapvalidminustrain", "mean"),
        )
        .sort_values(
            ["cvvalidwamean", "cvvalidwastd", "cvvalidrmsemean"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    best = cvsummary.iloc[0].to_dict()
    return cvdetail, cvsummary, best


def buildclusterprofiles(traindf: pd.DataFrame, labels: np.ndarray, featurecols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    profdf = traindf[[TIME_COL, TARGET_COL] + featurecols].copy()
    profdf["cluster"] = labels
    clustersizes = profdf.groupby("cluster").size().reset_index(name="size")
    clustersizes["share"] = clustersizes["size"] / len(profdf)
    targetstats = profdf.groupby("cluster")[TARGET_COL].agg(["mean", "median", "std", "min", "max"]).reset_index()
    targetstats = targetstats.rename(columns={"mean": "targetmean", "median": "targetmedian", "std": "targetstd", "min": "targetmin", "max": "targetmax"})
    targetstats["targetpositiverate"] = (
        profdf.assign(targetpositive=(profdf[TARGET_COL] > 0).astype(float))
        .groupby("cluster")["targetpositive"]
        .mean()
        .values
    )
    featuremeans = profdf.groupby("cluster")[featurecols].mean().reset_index()
    globalmeans = traindf[featurecols].mean(numeric_only=True)
    globalstds = traindf[featurecols].std(ddof=0, numeric_only=True).replace(0.0, 1.0).fillna(1.0)
    zrows = []
    for _, row in featuremeans.iterrows():
        cluster = int(row["cluster"])
        vals = row[featurecols]
        z = ((vals - globalmeans) / globalstds).to_dict()
        z["cluster"] = cluster
        zrows.append(z)
    profilezscores = pd.DataFrame(zrows)
    return clustersizes, targetstats, profilezscores


def saveclustercharts(Xtrainscaled: np.ndarray, labels: np.ndarray, targetstats: pd.DataFrame, clustersizes: pd.DataFrame, profilezscores: pd.DataFrame, outdir: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Xtrainscaled)
    pcadf = pd.DataFrame({"pc1": coords[:, 0], "pc2": coords[:, 1], "cluster": labels.astype(str)})
    fig1 = px.scatter(pcadf, x="pc1", y="pc2", color="cluster", title="Clusters KMeans en PCA (train)", opacity=0.65)
    fig1.update_xaxes(title_text="PC1")
    fig1.update_yaxes(title_text="PC2")
    fig1.write_image(str(outdir / "kmeans_clusters_pca.png"))
    with open(outdir / "kmeans_clusters_pca.png.meta.json", "w", encoding="utf-8") as f:
        json.dump({"caption": "Projection PCA des clusters KMeans", "description": "Nuage PCA du train d'estimation, coloré par cluster KMeans."}, f)

    fig2 = px.bar(targetstats.sort_values("targetmean"), x="cluster", y="targetmean", title="Moyenne de la cible par cluster")
    fig2.update_xaxes(title_text="Cluster")
    fig2.update_yaxes(title_text="Delta moyen")
    fig2.write_image(str(outdir / "kmeans_cluster_target_mean.png"))

    fig3 = px.bar(clustersizes.sort_values("cluster"), x="cluster", y="size", title="Taille des clusters KMeans")
    fig3.update_xaxes(title_text="Cluster")
    fig3.update_yaxes(title_text="Taille")
    fig3.write_image(str(outdir / "kmeans_cluster_sizes.png"))

    heatmapdf = profilezscores.set_index("cluster")
    topfeatures = heatmapdf.abs().mean(axis=0).sort_values(ascending=False).head(12).index.tolist()
    heatmapdf = heatmapdf[topfeatures]
    fig4 = go.Figure(data=go.Heatmap(z=heatmapdf.values, x=heatmapdf.columns.tolist(), y=[str(idx) for idx in heatmapdf.index.tolist()], colorscale="RdBu", zmid=0))
    fig4.update_layout(title="Signatures normalisées des clusters")
    fig4.update_xaxes(title_text="Features")
    fig4.update_yaxes(title_text="Cluster")
    fig4.write_image(str(outdir / "kmeans_cluster_signatures.png"))


def parseargs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entraîne un KMeans optimisé par CV temporelle.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"), help="Dossier des données préparées")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="Dossier des sorties")
    parser.add_argument("--k-min", type=int, default=2, help="Plus petite valeur de k testée")
    parser.add_argument("--k-max", type=int, default=20, help="Plus grande valeur de k testée")
    parser.add_argument("--cv-splits", type=int, default=3, help="Nombre de splits pour la validation croisée temporelle")
    parser.add_argument("--init", type=str, default="k-means++", help="Initialisation KMeans par défaut")
    parser.add_argument("--n-init", type=int, default=20, help="Nombre de redémarrages KMeans par défaut")
    parser.add_argument("--init-grid", nargs="*", default=None, help="Grille d'initialisations, ex: --init-grid k-means++ random")
    parser.add_argument("--n-init-grid", nargs="*", type=int, default=None, help="Grille de n_init, ex: --n-init-grid 10 20 50")
    parser.add_argument("--random-state", type=int, default=42, help="Graine aléatoire")
    return parser.parse_args()


def main() -> None:
    args = parseargs()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traindf = pd.read_csv(args.data_dir / "train_estimation.csv")
    validdf = pd.read_csv(args.data_dir / "train_validation.csv")
    testdf = pd.read_csv(args.data_dir / "test_features.csv")

    kvalues = list(range(args.k_min, args.k_max + 1))
    initgrid = expand_init_grid(args.init_grid, args.init)
    ninitgrid = expand_n_init_grid(args.n_init_grid, args.n_init)

    cvdetail, cvsummary, bestcfg = timeseriescvsearch(
        traindf=traindf,
        kvalues=kvalues,
        nsplits=args.cv_splits,
        initgrid=initgrid,
        ninitgrid=ninitgrid,
        randomstate=args.random_state,
    )

    bestk = int(bestcfg["k"])
    bestinit = str(bestcfg["init"])
    bestninit = int(bestcfg["ninit"])

    model = fitkmeansmodel(traindf=traindf, k=bestk, init=bestinit, ninit=bestninit, randomstate=args.random_state)
    trainpred, trainlabels = predictwithmodel(traindf, model)
    validpred, _ = predictwithmodel(validdf, model)
    testpred, _ = predictwithmodel(testdf, model)

    ytrain = traindf[TARGET_COL].to_numpy(dtype=float)
    yvalid = validdf[TARGET_COL].to_numpy(dtype=float)
    yhattrain = trainpred[TARGET_COL].to_numpy(dtype=float)
    yhatvalid = validpred[TARGET_COL].to_numpy(dtype=float)
    trainwa, trainrmse, trainmae = regressionmetrics(ytrain, yhattrain)
    validwa, validrmse, validmae = regressionmetrics(yvalid, yhatvalid)

    featurecols = model.preprocessor.featurecols
    clustersizes, targetstats, profilezscores = buildclusterprofiles(traindf, trainlabels, featurecols)
    Xtrainscaled = transformfeatures(traindf, model.preprocessor)

    cvdetail.to_csv(args.out_dir / "kmeans_cv_grid.csv", index=False)
    cvsummary.to_csv(args.out_dir / "kmeans_cv_summary.csv", index=False)
    trainpred.to_csv(args.out_dir / "kmeans_train_predictions.csv", index=False)
    validpred[[TIME_COL, TARGET_COL, "cluster"]].to_csv(args.out_dir / "kmeans_valid_predictions.csv", index=False)
    testpred[[TIME_COL, TARGET_COL, "cluster"]].to_csv(args.out_dir / "kmeans_test_submission.csv", index=False)
    clustersizes.to_csv(args.out_dir / "kmeans_cluster_sizes.csv", index=False)
    targetstats.to_csv(args.out_dir / "kmeans_cluster_target_stats.csv", index=False)
    profilezscores.to_csv(args.out_dir / "kmeans_cluster_profile_zscores.csv", index=False)

    topfeatures = profilezscores.set_index("cluster").abs().mean(axis=0).sort_values(ascending=False).head(8).index.tolist()
    topsignaturerows = []
    for clusterid in sorted(profilezscores["cluster"].tolist()):
        row = profilezscores[profilezscores["cluster"] == clusterid].iloc[0]
        ranked = sorted([(feat, float(row[feat])) for feat in topfeatures], key=lambda x: abs(x[1]), reverse=True)[:4]
        desc = ", ".join(f"{feat}={val:+.2f}σ" for feat, val in ranked)
        topsignaturerows.append({"cluster": clusterid, "signature_top_features": desc})
    signaturedf = pd.DataFrame(topsignaturerows)
    signaturedf.to_csv(args.out_dir / "kmeans_cluster_signature_summary.csv", index=False)

    report = f"""# Modèle non supervisé : KMeans optimisé par validation croisée temporelle

Clustering KMeans sur les variables numériques standardisées, avec mapping cluster -> moyenne de la cible observée sur l'échantillon d'estimation.

k retenu par CV temporelle: {bestk}

Initialisation retenue: {bestinit}

n_init retenu: {bestninit}

Validation croisée temporelle: {args.cv_splits} splits

Taille de grille évaluée: {len(kvalues)} x {len(initgrid)} x {len(ninitgrid)} = {len(kvalues) * len(initgrid) * len(ninitgrid)} configurations

Critère de sélection robuste: WA validation moyenne DESC, WA std ASC, RMSE validation moyen ASC

Inertie du modèle final: {model.kmeans.inertia_:.6f}

Train WA: {trainwa:.6f}

Train RMSE: {trainrmse:.6f}

Train MAE: {trainmae:.6f}

Validation WA: {validwa:.6f}

Validation RMSE: {validrmse:.6f}

Validation MAE: {validmae:.6f}

Écart moyen train-validation WA en CV: {bestcfg['cvwagapmean']:.6f}

Écart-type WA validation en CV: {bestcfg['cvvalidwastd']:.6f}

Conclusion sur le sur-apprentissage: {('gap faible, pas de signal fort de sur-apprentissage' if abs(float(bestcfg['cvwagapmean'])) < 0.05 else 'gap non négligeable, à commenter avec prudence')}

## Résumé CV

```csv
{cvsummary.to_csv(index=False)}
```

## Signatures courtes des clusters

```csv
{signaturedf.to_csv(index=False)}
```
"""
    (args.out_dir / "kmeans_metrics.md").write_text(report, encoding="utf-8")

    try:
        saveclustercharts(Xtrainscaled, trainlabels, targetstats, clustersizes, profilezscores, args.out_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()