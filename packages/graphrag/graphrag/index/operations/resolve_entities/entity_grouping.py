# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Union-Find based grouping of similar entity candidates."""


class _UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def build_candidate_groups(
    titles: list[str],
    embeddings: list[list[float]],
    threshold: float,
) -> list[list[int]]:
    """Return groups of entity indices whose embeddings exceed the similarity threshold.

    Uses cosine similarity (dot product of normalised vectors) and Union-Find
    for transitive closure: if A~B and B~C, all three end up in one group.

    Only groups with at least 2 members are returned — single-entity groups
    are already unique and need no resolution.

    Parameters
    ----------
    titles : list[str]
        Entity titles in the same order as ``embeddings``.
    embeddings : list[list[float]]
        One embedding vector per entity.
    threshold : float
        Minimum cosine similarity to consider two entities as merge candidates.

    Returns
    -------
    list[list[int]]
        Each inner list contains the indices (into ``titles``) of one merge group.
    """
    import numpy as np

    n = len(titles)
    if n < 2:
        return []

    mat = np.array(embeddings, dtype="float32")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Avoid division by zero for zero-vectors
    norms = np.where(norms == 0, 1.0, norms)
    mat = mat / norms

    uf = _UnionFind(n)

    # Chunked comparison to avoid building a full N×N matrix for large corpora
    chunk = 256
    for i in range(0, n, chunk):
        sims = mat[i : i + chunk] @ mat.T  # shape: (chunk, n)
        rows, cols = np.where(sims >= threshold)
        for r, c in zip(rows, cols, strict=False):
            global_i = i + int(r)
            global_j = int(c)
            if global_i != global_j:
                uf.union(global_i, global_j)

    # Collect groups
    from collections import defaultdict

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        groups[uf.find(idx)].append(idx)

    return [members for members in groups.values() if len(members) >= 2]
