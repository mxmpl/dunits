import csv
import re
from os import PathLike
from pathlib import Path

import numpy as np
from tabulate import tabulate

__all__ = ["read_units", "read_alignments", "count_matrix", "proba_phone_code", "units_quality"]


def read_manifest(file_path: PathLike) -> tuple[list[Path], list[int]]:
    filenames, num_samples = [], []
    with open(file_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        root = Path(next(reader)[0])
        for row in reader:
            assert len(row) == 2, f"Invalid tsv file: {file_path}"
            filenames.append(root / row[0])
            num_samples.append(int(row[1]))
    return filenames, num_samples


def read_units(units_path: PathLike, manifest_path: PathLike, sep: str = " ") -> dict[str, list[int]]:
    with open(units_path, "r") as f:
        lines = f.read().splitlines()
    filenames = read_manifest(manifest_path)[0]
    assert len(lines) == len(filenames)
    units = {file.stem: [int(unit) for unit in line.split(sep)] for line, file in zip(lines, filenames)}
    assert len(units) == len(lines)
    return units


def read_alignments(alignments_path: PathLike, sep: str = ",") -> dict[str, list[str]]:
    phones = {}
    with open(alignments_path, "r", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            assert len(row) == 2
            phones[row[0]] = row[1].split(sep)
    return phones


def count_matrix(
    phones: dict[str, list[str]], units: dict[str, list[int]], repeat: int = 2
) -> tuple[np.ndarray, list[str]]:
    num_phones = len(np.unique([re.sub("[0-9]", "", p) for phone in phones.values() for p in phone]))
    num_units = max([max(unit) for unit in units.values()]) + 1
    count = np.zeros((num_phones, num_units), dtype=int)

    dictionnary, idx = {}, 0
    for fileid, unit in units.items():
        unit = np.repeat(unit, repeat)
        phone = phones[fileid]
        assert abs(len(phone) - len(unit)) <= 1, (len(phones), len(units), fileid)
        for p, u in zip(phone, unit):
            p = re.sub("[0-9]", "", p)
            if p not in dictionnary:
                dictionnary[p] = idx
                idx += 1
            count[dictionnary[p], u] += 1

    most_frequent_phones = np.argsort(count.sum(axis=1))[::-1]
    phone_order = [{v: k for k, v in dictionnary.items()}[idx] for idx in most_frequent_phones]
    count = count[most_frequent_phones]
    return count, phone_order


def proba_phone_code(count: np.ndarray) -> tuple[np.ndarray, list[int]]:
    count_by_code = count.sum(axis=0, keepdims=True)
    proba = np.divide(count, count_by_code, out=np.zeros_like(count, dtype="float64"), where=count_by_code != 0)
    assert not np.any(np.isnan(proba))
    units_order, argmax = [], proba.argmax(axis=0)
    for phone_index in range(len(count)):
        indices = np.where(argmax == phone_index)[0]
        units_order.extend(indices[np.argsort(proba[phone_index, indices])[::-1]])
    return proba[:, units_order], units_order


def phone_purity(proba: np.ndarray) -> float:
    return proba.max(axis=0).sum()


def cluster_purity(proba: np.ndarray) -> float:
    return proba.max(axis=1).sum()


def pnmi(proba: np.ndarray) -> float:
    px = proba.sum(axis=1, keepdims=True)
    py = proba.sum(axis=0, keepdims=True)
    mutual_info = (proba * np.log(proba / (px @ py + 1e-8) + 1e-8)).sum()
    entropy_x = (-px * np.log(px + 1e-8)).sum()
    return mutual_info / entropy_x


def units_quality(count: np.ndarray) -> None:
    proba = count / count.sum()
    print(
        tabulate(
            [
                ["Phone purity", phone_purity(proba)],
                ["Cluster purity", cluster_purity(proba)],
                ["PNMI", pnmi(proba)],
            ]
        )
    )
