"""Shared plant metadata and lightweight auto-detection helpers."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class PlantConfig:
    key: str
    label: str
    scientific_name: str
    uniprot_tax_id: int
    string_species_id: int
    short_name: str
    patterns: List[str]
    min_hits: int = 5

    def compiled_patterns(self) -> List[re.Pattern]:
        return _COMPILED_PATTERNS[self.key]


PLANT_REGISTRY: Dict[str, PlantConfig] = {
    "rice": PlantConfig(
        key="rice",
        label="Rice",
        scientific_name="Oryza sativa",
        uniprot_tax_id=39947,
        string_species_id=39947,
        short_name="Rice",
        patterns=[r"^LOC_Os\d{2}g", r"^Os\d", r"^Os[A-Z]+"],
    ),
    "arabidopsis": PlantConfig(
        key="arabidopsis",
        label="Arabidopsis",
        scientific_name="Arabidopsis thaliana",
        uniprot_tax_id=3702,
        string_species_id=3702,
        short_name="Arabidopsis",
        patterns=[r"^AT[1-5CM]G\d+", r"^AT\d+G", r"^ATH"],
        min_hits=8,
    ),
    "maize": PlantConfig(
        key="maize",
        label="Maize",
        scientific_name="Zea mays",
        uniprot_tax_id=4577,
        string_species_id=4577,
        short_name="Maize",
        patterns=[r"^GRMZM\d", r"^Zm\d", r"^Zm[0-9A-Z]{4}"],
    ),
    "wheat": PlantConfig(
        key="wheat",
        label="Wheat",
        scientific_name="Triticum aestivum",
        uniprot_tax_id=4565,
        string_species_id=4565,
        short_name="Wheat",
        patterns=[r"^TraesCS", r"^Ta\d", r"^Tae?\d"],
    ),
    "soybean": PlantConfig(
        key="soybean",
        label="Soybean",
        scientific_name="Glycine max",
        uniprot_tax_id=3847,
        string_species_id=3847,
        short_name="Soybean",
        patterns=[r"^Glyma\d", r"^GM[0-9A-Z]{4}"],
    ),
    "barley": PlantConfig(
        key="barley",
        label="Barley",
        scientific_name="Hordeum vulgare",
        uniprot_tax_id=4513,
        string_species_id=4513,
        short_name="Barley",
        patterns=[r"^HORVU", r"^Hv\d"],
    ),
    "sorghum": PlantConfig(
        key="sorghum",
        label="Sorghum",
        scientific_name="Sorghum bicolor",
        uniprot_tax_id=4558,
        string_species_id=4558,
        short_name="Sorghum",
        patterns=[r"^Sobic", r"^Sb\d"],
    ),
    "brachypodium": PlantConfig(
        key="brachypodium",
        label="Brachypodium",
        scientific_name="Brachypodium distachyon",
        uniprot_tax_id=15368,
        string_species_id=15368,
        short_name="Brachypodium",
        patterns=[r"^BRADI", r"^Bd\d"],
    ),
    "tomato": PlantConfig(
        key="tomato",
        label="Tomato",
        scientific_name="Solanum lycopersicum",
        uniprot_tax_id=4081,
        string_species_id=4081,
        short_name="Tomato",
        patterns=[r"^Solyc", r"^Sl\d"],
    ),
    "potato": PlantConfig(
        key="potato",
        label="Potato",
        scientific_name="Solanum tuberosum",
        uniprot_tax_id=4113,
        string_species_id=4113,
        short_name="Potato",
        patterns=[r"^PGSC", r"^St\d"],
    ),
}

_COMPILED_PATTERNS: Dict[str, List[re.Pattern]] = {
    key: [re.compile(ptrn, re.IGNORECASE) for ptrn in cfg.patterns]
    for key, cfg in PLANT_REGISTRY.items()
}

DEFAULT_SPECIES_KEY = "rice"


def list_species_options() -> List[Dict[str, str]]:
    return [
        {
            "key": cfg.key,
            "label": f"{cfg.label} ({cfg.scientific_name})",
            "short": cfg.short_name,
            "scientific_name": cfg.scientific_name,
            "uniprot_tax_id": cfg.uniprot_tax_id,
            "string_species_id": cfg.string_species_id,
            "display_name": cfg.label,
            "image_path": f"/static/media/images/species/{cfg.key}.png",
        }
        for cfg in PLANT_REGISTRY.values()
    ]


def get_species_config(key: Optional[str]) -> PlantConfig:
    if not key:
        return PLANT_REGISTRY[DEFAULT_SPECIES_KEY]
    return PLANT_REGISTRY.get(key, PLANT_REGISTRY[DEFAULT_SPECIES_KEY])


def detect_species_from_genes(
    genes: Iterable[str],
    min_ratio: float = 0.05,
    max_samples: int = 2000,
) -> Optional[str]:
    sample: List[str] = []
    for value in genes:
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        sample.append(text)
        if len(sample) >= max_samples:
            break

    if not sample:
        return None

    totals: Dict[str, int] = {cfg.key: 0 for cfg in PLANT_REGISTRY.values()}
    for gene in sample:
        for cfg in PLANT_REGISTRY.values():
            for ptrn in cfg.compiled_patterns():
                if ptrn.search(gene):
                    totals[cfg.key] += 1
                    break
            else:
                continue
            break

    best_key = max(totals, key=lambda k: totals[k])
    hits = totals[best_key]
    if hits == 0:
        return None

    ratio = hits / max(len(sample), 1)
    cfg = get_species_config(best_key)
    threshold = max(cfg.min_hits, int(len(sample) * min_ratio))
    if hits < threshold:
        return None
    return best_key


__all__ = [
    "DEFAULT_SPECIES_KEY",
    "PlantConfig",
    "PLANT_REGISTRY",
    "detect_species_from_genes",
    "get_species_config",
    "list_species_options",
]
