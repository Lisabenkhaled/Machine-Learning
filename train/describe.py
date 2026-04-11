from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_feature_stats(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["number"]).columns
    stats = df[numeric_cols].describe().T
    stats["missing_ratio"] = df[numeric_cols].isna().mean()
    stats = stats.sort_values("std", ascending=False)
    return stats.head(top_n)


def _df_to_markdown_no_tabulate(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Convertit un DataFrame en markdown sans dépendance externe (`tabulate`)."""
    cols = [str(c) for c in df.columns]

    def _format_value(v):
        if isinstance(v, float):
            return format(v, floatfmt)
        return str(v)

    header = "| feature | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"

    rows = []
    for idx, row in df.iterrows():
        values = " | ".join(_format_value(v) for v in row.tolist())
        rows.append(f"| {idx} | {values} |")

    return "\n".join([header, sep, *rows])


def write_markdown_report(stats: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Description des features")
    lines.append("")
    lines.append("Ce rapport résume les variables explicatives (numériques) les plus dispersées.")
    lines.append("")
    lines.append("## Statistiques descriptives")
    lines.append("")
    lines.append(_df_to_markdown_no_tabulate(stats, floatfmt=".4f"))
    lines.append("")
    lines.append("## Interprétation rapide")
    lines.append("")
    lines.append("- `load_forecast` : proxy direct de la demande; une demande élevée pousse souvent les prix à la hausse.")
    lines.append("- `gas_power_available` : le gaz est fréquemment marginal sur le marché, donc fortement lié au prix spot.")
    lines.append("- `wind/solar *_average` : plus la production ENR anticipée est forte, plus la pression baissière sur les prix est probable.")
    lines.append("- `*_std` : mesure l'incertitude des prévisions ENR; l'incertitude peut augmenter la volatilité intraday.")
    lines.append("- Variables calendaires (`hour`, `dayofweek`, sin/cos) : capturent les cycles journaliers et hebdomadaires.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
