"""The label-permutation null of a group's mAP: relabel profiles within their blocks, reuse their similarities.

Under the null hypothesis a group's members and the references they are ranked against come from one distribution,
so which of them carry the group's label is arbitrary. A draw moves the label, the group's mAP is recomputed from
the observed similarities with the group's positive graph between the relabelled slots, and the p-value counts the
draws whose mAP reaches the observed one. This is the null copairs calls ``permutation``; the implementation is not
copairs', and its draws are not copairs' draws.

Single label
    Members ranked against the same references form a block (a plate's replicates against that plate's controls).
    A draw fills each block's member slots with a random ordered subset of the block's pool, its members and their
    references. A slot's negatives are the pool minus the profiles drawn into the block.
Multilabel
    A label's blocks are the components of the graph of negative pairs that hold its profiles. A draw gives the
    label to as many of each component's profiles as hold it there. A slot's negatives are the profile's own
    negatives and the profiles that share only this label with it, minus the drawn profiles.

All a draw needs is, for each slot, how many of its negatives rank before each of its positives. Most of those
negatives form a fixed set per profile and block, so each (profile, set) is scored and sorted once, shared by every
group and draw that ranks against it, and a draw binary-searches it; only the handful of profiles a draw adds to or
takes out of that set are scored during the draw. Draw ``j`` of a group is a pure function of the seed, the group's
members and ``j``, so it does not depend on the threads or on the other groups.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from mantispy._core._average_precision import (
    BUILTIN,
    APResult,
    Layout,
    _similarity,
    is_negative,
    prepare_rows,
    summation_plan,
)
from mantispy._core._map_null import tie_thresholds
from mantispy._core._numba import N_CHUNKS

if TYPE_CHECKING:
    from builtins import range as prange
else:
    from numba import prange

#: Bytes of sorted keys the null may hold; beyond it the call refuses and names the nulls that need none.
STORE_BUDGET = 2**33
#: A block ranks against its whole ``neg_sameby`` bucket, which every group in the bucket shares, unless the bucket's
#: profiles that are neither its members nor its references outnumber its pool this many times over.
_MAX_EXPLICIT = 4
#: Bytes of the key tables between pool entries; groups beyond it score their keys during the draws.
_TABLE_BUDGET = 2**31
#: A pool entry without a subtracted set.
_NO_SET = -(2**62)
#: Draws per work item: enough to amortize a group's setup, few enough to spread one group over the threads.
_DRAW_CHUNK = 256
#: How a slot's fixed set relates to its negatives.
_REFERENCES, _BUCKET, _OWN = 0, 1, 2


@njit(parallel=True, nogil=True, cache=True)
def _set_rows(rows, scale, metric, leaf_lo, leaf_n, ops, owner, row_set, set_ptr, members, out_ptr, out, n_chunks):
    """Sorted keys ``float32(1) - similarity`` of profile ``owner[r]`` to set ``row_set[r]``, itself left out."""
    one = np.float32(1.0)
    for chunk in prange(n_chunks):
        stack = np.empty(64, np.float32)
        for r in range(chunk, owner.shape[0], n_chunks):
            x = owner[r]
            p = out_ptr[r]
            for t in range(set_ptr[row_set[r]], set_ptr[row_set[r] + 1]):
                z = members[t]
                if z != x:
                    out[p] = one - _similarity(rows, scale, x, z, metric, leaf_lo, leaf_n, ops, stack)
                    p += 1
            out[out_ptr[r] : p].sort()


@njit(parallel=True, nogil=True, cache=True)
def _own_rows(
    rows, scale, metric, leaf_lo, leaf_n, ops, owner, order, off, neg_bucket, neg_diff, lab_ptr, lab, out_ptr, out,
    n_chunks,
):  # fmt: skip
    """Sorted keys of profile ``owner[r]`` to its own negatives."""
    one = np.float32(1.0)
    for chunk in prange(n_chunks):
        stack = np.empty(64, np.float32)
        for r in range(chunk, owner.shape[0], n_chunks):
            x = owner[r]
            p = out_ptr[r]
            for t in range(off[neg_bucket[x]], off[neg_bucket[x] + 1]):
                z = order[t]
                if is_negative(x, z, neg_bucket, neg_diff, True, lab_ptr, lab):
                    out[p] = one - _similarity(rows, scale, x, z, metric, leaf_lo, leaf_n, ops, stack)
                    p += 1
            out[out_ptr[r] : p].sort()


@njit(nogil=True, cache=True)
def _components(order, off, neg_bucket, neg_diff, lab_ptr, lab):
    """Component of every profile in the graph of (multilabel) negative pairs, by breadth-first search."""
    n = neg_bucket.shape[0]
    comp = np.full(n, -1, np.int64)
    queue = np.empty(n, np.int64)
    n_comp = 0
    for start in range(n):
        if comp[start] >= 0:
            continue
        comp[start] = n_comp
        head, tail = 0, 1
        queue[0] = start
        while head < tail:
            x = queue[head]
            head += 1
            for t in range(off[neg_bucket[x]], off[neg_bucket[x] + 1]):
                z = order[t]
                if comp[z] < 0 and is_negative(x, z, neg_bucket, neg_diff, True, lab_ptr, lab):
                    comp[z] = n_comp
                    queue[tail] = z
                    tail += 1
        n_comp += 1
    return comp


@njit(nogil=True, cache=True, inline="always")
def _mix(z):
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(nogil=True, cache=True, inline="always")
def _next(state):
    """splitmix64: the advanced state and a double in ``[0, 1)``."""
    state = state + np.uint64(0x9E3779B97F4A7C15)
    return state, np.float64(_mix(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(nogil=True, cache=True, inline="always")
def _count_below(keys, lo, hi, key):
    a, b = lo, hi
    while a < b:
        mid = (a + b) >> 1
        if keys[mid] < key:
            a = mid + 1
        else:
            b = mid
    return a - lo


@njit(nogil=True, cache=True, inline="always")
def _key(e1, e2, tidx, table_at, size, table, rows, scale, profile, metric, leaf_lo, leaf_n, ops, stack):
    """Rank key between pool entries ``e1`` and ``e2``: from the group's key table if it has one, else scored."""
    if table_at >= 0:
        return table[table_at + tidx[e1] * size + tidx[e2]]
    return np.float32(1.0) - _similarity(rows, scale, profile[e1], profile[e2], metric, leaf_lo, leaf_n, ops, stack)


@njit(parallel=True, nogil=True, cache=True)
def _key_tables(
    rows, scale, metric, leaf_lo, leaf_n, ops, row_profile, cols, row_lo, row_size, row_at, table, n_chunks
):
    """Key table rows: row ``r`` holds the keys of ``row_profile[r]`` to ``cols[row_lo[r]:row_lo[r] + row_size[r]]``."""
    one = np.float32(1.0)
    for chunk in prange(n_chunks):
        stack = np.empty(64, np.float32)
        for r in range(chunk, row_profile.shape[0], n_chunks):
            x = row_profile[r]
            for c in range(row_size[r]):
                z = cols[row_lo[r] + c]
                table[row_at[r] + c] = (
                    one if z == x else one - _similarity(rows, scale, x, z, metric, leaf_lo, leaf_n, ops, stack)
                )


@njit(parallel=True, nogil=True, cache=True)
def _permutation_hits(
    rows, scale, metric, leaf_lo, leaf_n, ops, group_slot, group_block, group_key, thresholds, group_table,
    group_size, tidx, table, slot_block, adj_ptr, adj, block_slot, block_pool, block_kind, profile, entry_row, entry_sub_row, entry_lone,
    lone_ptr, lone, neg_bucket, neg_diff, lab_ptr, lab, store_ptr, store, items, max_slots, max_pool, max_scored,
    n_chunks,
):  # fmt: skip
    """Draws of each work item ``(group, first draw, end draw)`` whose mAP reaches the group's threshold.

    A group's slots are ``group_slot[g]:group_slot[g + 1]`` and its blocks ``group_block[g]:group_block[g + 1]``;
    block ``b``'s slots are ``block_slot[b]:block_slot[b + 1]``, in order, and its pool entries
    ``block_pool[b]:block_pool[b + 1]``, its members first. Each entry ranks against the sorted keys of
    ``entry_row``, less those of ``entry_sub_row`` (``-1`` for none). ``group_table[g]`` locates the group's key
    table in ``table`` (``-1`` when the keys are scored during the draws), ``group_size[g]`` its width and ``tidx``
    each entry's row and column in it.
    """
    hits = np.zeros(items.shape[0], np.int64)
    for chunk in prange(n_chunks):
        stack = np.empty(64, np.float32)
        slot_entry = np.empty(max_slots, np.int64)
        drawn = np.zeros(max_pool, np.bool_)
        chosen = np.empty(max_slots, np.int64)
        added = np.empty(max_scored, np.float32)
        removed = np.empty(max_scored, np.float32)
        pos_key = np.empty(max_slots, np.float32)
        pos_before = np.empty(max_slots, np.int64)
        for k in range(chunk, items.shape[0], n_chunks):
            g, d_lo, d_hi = items[k, 0], items[k, 1], items[k, 2]
            s0, s1 = group_slot[g], group_slot[g + 1]
            b0, b1 = group_block[g], group_block[g + 1]
            e0 = block_pool[b0]
            at = group_table[g]
            size = group_size[g]
            for j in range(d_lo, d_hi):
                state = _mix(group_key[g] ^ _mix(np.uint64(j) + np.uint64(0x632BE59BD9B4E019)))
                for b in range(b0, b1):  # Floyd's k-subset of the block's pool, then a uniform order
                    kk = block_slot[b + 1] - block_slot[b]
                    pp = block_pool[b + 1] - block_pool[b]
                    for c in range(kk):
                        t = pp - kk + c
                        state, u = _next(state)
                        r = min(np.int64(u * (t + 1)), t)
                        for z in range(c):
                            if chosen[z] == r:
                                r = t
                                break
                        chosen[c] = r
                    for z in range(kk - 1, 0, -1):
                        state, u = _next(state)
                        r = min(np.int64(u * (z + 1)), z)
                        chosen[z], chosen[r] = chosen[r], chosen[z]
                    for c in range(kk):
                        e = block_pool[b] + chosen[c]
                        slot_entry[block_slot[b] + c - s0] = e
                        drawn[e - e0] = True
                total = 0.0
                for s in range(s0, s1):
                    e = slot_entry[s - s0]
                    b = slot_block[s]
                    n_add = 0
                    n_rem = 0
                    kind = block_kind[b]
                    if kind == 2:  # fixed set: x's own negatives; undrawn lone partners join, drawn negatives leave
                        lx = entry_lone[e]
                        if lx >= 0:
                            for t in range(lone_ptr[lx], lone_ptr[lx + 1]):
                                ze = lone[t]
                                if not drawn[ze - e0]:
                                    added[n_add] = _key(
                                        e,
                                        ze,
                                        tidx,
                                        at,
                                        size,
                                        table,
                                        rows,
                                        scale,
                                        profile,
                                        metric,
                                        leaf_lo,
                                        leaf_n,
                                        ops,
                                        stack,
                                    )
                                    n_add += 1
                        for t in range(s0, s1):
                            ze = slot_entry[t - s0]
                            if ze != e and is_negative(
                                profile[e], profile[ze], neg_bucket, neg_diff, True, lab_ptr, lab
                            ):
                                removed[n_rem] = _key(
                                    e,
                                    ze,
                                    tidx,
                                    at,
                                    size,
                                    table,
                                    rows,
                                    scale,
                                    profile,
                                    metric,
                                    leaf_lo,
                                    leaf_n,
                                    ops,
                                    stack,
                                )
                                n_rem += 1
                    else:
                        n_mem = block_slot[b + 1] - block_slot[b]
                        if kind == 0:  # fixed set: the block's references; its undrawn members join
                            for ze in range(block_pool[b], block_pool[b] + n_mem):
                                if ze != e and not drawn[ze - e0]:
                                    added[n_add] = _key(
                                        e,
                                        ze,
                                        tidx,
                                        at,
                                        size,
                                        table,
                                        rows,
                                        scale,
                                        profile,
                                        metric,
                                        leaf_lo,
                                        leaf_n,
                                        ops,
                                        stack,
                                    )
                                    n_add += 1
                        for t in range(block_slot[b], block_slot[b + 1]):  # the other drawn profiles leave
                            ze = slot_entry[t - s0]
                            if ze != e and (kind == 1 or ze >= block_pool[b] + n_mem):
                                removed[n_rem] = _key(
                                    e,
                                    ze,
                                    tidx,
                                    at,
                                    size,
                                    table,
                                    rows,
                                    scale,
                                    profile,
                                    metric,
                                    leaf_lo,
                                    leaf_n,
                                    ops,
                                    stack,
                                )
                                n_rem += 1
                    lo, hi = store_ptr[entry_row[e]], store_ptr[entry_row[e] + 1]
                    sub = entry_sub_row[e]
                    deg = 0
                    for t in range(adj_ptr[s], adj_ptr[s + 1]):
                        key = _key(
                            e, slot_entry[adj[t] - s0], tidx, at, size, table, rows, scale, profile, metric, leaf_lo,
                            leaf_n, ops, stack,
                        )  # fmt: skip
                        # ties rank the positive first, so only negatives strictly before it count
                        before = _count_below(store, lo, hi, key)
                        if sub >= 0:
                            before -= _count_below(store, store_ptr[sub], store_ptr[sub + 1], key)
                        for q in range(n_add):
                            before += added[q] < key
                        for q in range(n_rem):
                            before -= removed[q] < key
                        q = deg
                        while q > 0 and pos_key[q - 1] > key:
                            pos_key[q] = pos_key[q - 1]
                            pos_before[q] = pos_before[q - 1]
                            q -= 1
                        pos_key[q] = key
                        pos_before[q] = before
                        deg += 1
                    acc = 0.0
                    for q in range(deg):
                        acc += (q + 1) / (q + 1 + pos_before[q])
                    total += acc / deg
                if total / (s1 - s0) >= thresholds[g]:
                    hits[k] += 1
                for s in range(s0, s1):
                    drawn[slot_entry[s - s0] - e0] = False
    return hits


def _group_key(seed: int, members: np.ndarray, label: int) -> np.uint64:
    text = repr((int(seed), tuple(sorted(int(m) for m in members)), int(label)))
    return np.uint64(int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"))


def _refuse(message: str) -> None:
    raise ValueError(f"{message} Pass null='copied' or null='joint', which need no relabelling.")


class _Plan:
    """Flat arrays of every group's slots, blocks and pool entries, and the fixed sets the entries rank against."""

    def __init__(self) -> None:
        self.group_slot: list[int] = [0]
        self.group_block: list[int] = [0]
        self.group_key: list[np.uint64] = []
        self.slot_block: list[int] = []
        self.adj_ptr: list[int] = [0]
        self.adj: list[int] = []
        self.block_slot: list[int] = [0]
        self.block_pool: list[int] = [0]
        self.block_kind: list[int] = []
        self.entry_profile: list[int] = []
        self.entry_set: list[int] = []
        self.entry_sub_set: list[int] = []
        self.entry_lone: list[int] = []
        self.lone_ptr: list[int] = [0]
        self.lone: list[int] = []
        self.sets: list[np.ndarray] = []
        self._set_index: dict[bytes, int] = {}

    def set_id(self, members: np.ndarray) -> int:
        """Index of a fixed set of profiles; ``-1 - p`` stands for profile ``p``'s own negatives."""
        members = np.sort(np.asarray(members, dtype=np.int64))
        key = members.tobytes()
        if key not in self._set_index:
            self._set_index[key] = len(self.sets)
            self.sets.append(members)
        return self._set_index[key]

    def add_group(self, key, slots_adj, blocks):
        """Append one group.

        ``blocks`` lists per block its slots (local), pool entries (members first), kind, and each entry's fixed set,
        subtracted set (:data:`_NO_SET` for none) and lone index.
        """
        s0 = self.group_slot[-1]
        first_block = len(self.block_kind)
        slot_block = np.empty(len(slots_adj), np.int64)
        for b, (slots, pool, kind, entry_set, entry_sub_set, entry_lone) in enumerate(blocks):
            slot_block[slots] = first_block + b
            self.block_slot.append(self.block_slot[-1] + len(slots))
            self.entry_profile.extend(int(p) for p in pool)
            self.entry_set.extend(entry_set)
            self.entry_sub_set.extend(entry_sub_set)
            self.entry_lone.extend(entry_lone)
            self.block_pool.append(len(self.entry_profile))
            self.block_kind.append(kind)
        # slots are numbered block by block, so a block's slots are contiguous
        order = np.concatenate([np.asarray(slots, dtype=np.int64) for slots, *_ in blocks])
        new_of_old = np.empty(len(order), np.int64)
        new_of_old[order] = np.arange(len(order))
        for old in order:
            self.slot_block.append(int(slot_block[old]))
            self.adj.extend(int(s0 + new_of_old[t]) for t in slots_adj[old])
            self.adj_ptr.append(len(self.adj))
        self.group_slot.append(s0 + len(order))
        self.group_block.append(len(self.block_kind))
        self.group_key.append(key)


def permutation_pvalues(
    lay: Layout,
    result: APResult,
    features: np.ndarray,
    distance: str,
    scores: np.ndarray,
    groups: list[np.ndarray],
    rows: np.ndarray,
    null_size: int,
    seed: int,
    n_chunks: int = N_CHUNKS,
) -> np.ndarray:
    """P-values of the groups' mAPs against relabellings of their profiles.

    ``groups[g]`` lists positions of the tested table and ``rows`` maps them to AP rows of ``result``.

    Raises:
        ValueError: A group's positives leave it, a negative joins two of its members, members of one group rank
            against overlapping but different references, a multilabel group does not hold every profile of its
            label, the distance has no compiled similarity, or the sorted keys would exceed :data:`STORE_BUDGET`.
    """
    if distance not in BUILTIN:
        raise ValueError(f"null='permutation' supports distance in {sorted(BUILTIN)}, got {distance!r}")
    x, scale = prepare_rows(features, distance)
    leaf_lo, leaf_n, ops = summation_plan(x.shape[1])
    metric = BUILTIN[distance]
    order = np.argsort(lay.neg_bucket, kind="stable").astype(np.int64)
    off = np.searchsorted(lay.neg_bucket[order], np.arange(lay.n_buckets + 1)).astype(np.int64)
    n_labels = int(result.row_label.max(initial=0)) + 1
    row_key = result.row_profile * n_labels + result.row_label
    comp = (
        _components(order, off, lay.neg_bucket, lay.neg_diff, lay.lab_ptr, lay.lab)
        if lay.disjoint
        else np.zeros(lay.n, np.int64)
    )

    plan = _Plan()
    for ix in groups:
        ap_rows = rows[ix]
        members = result.row_profile[ap_rows]
        label = int(result.row_label[ap_rows[0]])
        local = {int(p): k for k, p in enumerate(members)}
        slots_adj = []
        for r in ap_rows:
            partners = result.partner[result.pos_ptr[r] : result.pos_ptr[r + 1]]
            if lay.disjoint:  # a positive of a multilabel row is a row of the same label
                found = np.isin(partners * n_labels + label, row_key)
            else:
                found = np.ones(len(partners), bool)
            if not found.all() or any(int(p) not in local for p in partners):
                _refuse(
                    "The permutation null needs each group's positives to be its other members, and a group has "
                    "a positive outside it (for example a member dropped for having no negative pair)."
                )
            slots_adj.append([local[int(p)] for p in partners])
        if lay.disjoint:
            blocks = _label_blocks(lay, plan, members, label, comp)
        else:
            blocks = _single_label_blocks(lay, plan, members, order, off)
        plan.add_group(_group_key(seed, members, label if lay.disjoint else -1), slots_adj, blocks)

    store_ptr, store, entry_row, entry_sub_row = _store(
        lay, result, plan, x, scale, metric, leaf_lo, leaf_n, ops, order, off, n_chunks
    )
    as64 = lambda a: np.asarray(a, dtype=np.int64)
    group_slot, group_block, block_pool = as64(plan.group_slot), as64(plan.group_block), as64(plan.block_pool)
    profile = as64(plan.entry_profile)
    group_table, group_size, tidx, table = _tables(
        plan, group_slot, group_block, block_pool, profile, null_size, x, scale, metric, leaf_lo, leaf_n, ops, n_chunks
    )
    items = np.array(
        [(g, d, min(d + _DRAW_CHUNK, null_size)) for g in range(len(groups)) for d in range(0, null_size, _DRAW_CHUNK)],
        dtype=np.int64,
    ).reshape(-1, 3)
    pool_size = block_pool[group_block[1:]] - block_pool[group_block[:-1]]
    max_slots = int(np.diff(group_slot).max(initial=1))
    max_scored = 2 * max_slots + int(np.diff(plan.lone_ptr).max(initial=0)) + 1
    hits = _permutation_hits(
        x, scale, metric, leaf_lo, leaf_n, ops, group_slot, group_block, np.array(plan.group_key, dtype=np.uint64),
        tie_thresholds(np.asarray(scores, dtype=np.float64)), group_table, group_size, tidx, table, as64(plan.slot_block),
        as64(plan.adj_ptr), as64(plan.adj), as64(plan.block_slot), block_pool, as64(plan.block_kind), profile,
        entry_row, entry_sub_row, as64(plan.entry_lone), as64(plan.lone_ptr), as64(plan.lone), lay.neg_bucket,
        lay.neg_diff, lay.lab_ptr, lay.lab, store_ptr, store, items, max_slots, int(pool_size.max(initial=1)),
        max_scored, max(1, min(n_chunks, len(items))),
    )  # fmt: skip
    per_group = np.bincount(items[:, 0], weights=hits, minlength=len(groups))
    return (per_group + 1) / (null_size + 1)


def _tables(plan, group_slot, group_block, block_pool, profile, null_size, x, scale, metric, leaf_lo, leaf_n, ops,
            n_chunks):  # fmt: skip
    """Key tables, for the groups whose draws would score more pairs than a table holds.

    One table over every profile of every pool when that is smaller than the groups' own tables together (groups
    whose pools are whole plates share them), else a table per group between its pool entries. Returns each group's
    offset into the tables (``-1`` for none) and width, each entry's row and column, and the tables.
    """
    n_groups = len(group_slot) - 1
    adj_ptr = np.asarray(plan.adj_ptr, dtype=np.int64)
    e0 = block_pool[group_block[:-1]]
    size = block_pool[group_block[1:]] - e0
    n_slots = np.diff(group_slot)
    scored = null_size * (adj_ptr[group_slot[1:]] - adj_ptr[group_slot[:-1]] + 2 * n_slots * n_slots)
    own = np.minimum(size * size, scored)
    union, position = np.unique(profile, return_inverse=True)
    rows_spec: tuple[np.ndarray, ...]
    if 4 * len(union) ** 2 <= _TABLE_BUDGET and len(union) ** 2 < own.sum():
        group_table = np.zeros(n_groups, np.int64)
        group_size = np.full(n_groups, len(union), np.int64)
        tidx = position.astype(np.int64)
        n = len(union)
        rows_spec = (union, union, np.zeros(n, np.int64), np.full(n, n, np.int64), np.arange(n, dtype=np.int64) * n)
        used = n * n
    else:
        group_table = np.full(n_groups, -1, np.int64)
        group_size = size.astype(np.int64)
        tidx = np.arange(len(profile), dtype=np.int64) - np.repeat(e0, size)
        used = 0
        owners, lo, width, at = [], [], [], []
        for g in np.flatnonzero(size * size <= scored):
            if 4 * (used + int(size[g]) ** 2) > _TABLE_BUDGET:
                break
            group_table[g] = used
            owners.append(np.arange(e0[g], e0[g] + size[g]))
            lo.append(np.full(size[g], e0[g]))
            width.append(np.full(size[g], size[g]))
            at.append(used + np.arange(size[g]) * size[g])
            used += int(size[g]) ** 2
        cat = lambda parts: np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, np.int64)
        rows_spec = (profile[cat(owners)], profile, cat(lo), cat(width), cat(at))
    table = np.empty(used, np.float32)
    if len(rows_spec[0]):
        _key_tables(x, scale, metric, leaf_lo, leaf_n, ops, *rows_spec, table, max(1, min(n_chunks, len(rows_spec[0]))))
    return group_table, group_size, tidx, table


def _single_label_blocks(lay: Layout, plan: _Plan, members: np.ndarray, order: np.ndarray, off: np.ndarray):
    """A group's blocks: members ranked against the same references, each with its pool and fixed sets."""
    signature = [(int(lay.neg_bucket[m]), tuple(lay.neg_diff[m].tolist())) for m in members]
    blocks = []
    seen: dict[int, np.ndarray] = {}
    member_set = {int(m) for m in members}
    for key in sorted(set(signature)):
        slots = [k for k, s in enumerate(signature) if s == key]
        bucket = order[off[key[0]] : off[key[0] + 1]]
        codes = np.asarray(key[1], dtype=np.int64)
        refs = bucket[(lay.neg_diff[bucket] != codes).all(axis=1)] if len(codes) else bucket
        if member_set & set(refs.tolist()):
            _refuse("The permutation null needs each group's references outside it, and a negative joins two members.")
        if key[0] in seen and len(np.intersect1d(seen[key[0]], refs)):
            _refuse("Members of a group rank against overlapping but different references.")
        seen[key[0]] = refs if key[0] not in seen else np.union1d(seen[key[0]], refs)
        pool = np.concatenate([members[slots], refs])
        outsiders = np.setdiff1d(bucket, pool)
        if len(outsiders) <= _MAX_EXPLICIT * len(pool):
            # rank against the whole bucket, which every group in it shares, less the bucket's other profiles
            kind, set_id = _BUCKET, plan.set_id(bucket)
            sub_id = plan.set_id(outsiders) if len(outsiders) else _NO_SET
        else:
            kind, set_id, sub_id = _REFERENCES, plan.set_id(refs), _NO_SET
        blocks.append((slots, pool, kind, [set_id] * len(pool), [sub_id] * len(pool), [-1] * len(pool)))
    return blocks


def _label_blocks(lay: Layout, plan: _Plan, members: np.ndarray, label: int, comp: np.ndarray):
    """A multilabel group's blocks: the components of the negative graph that hold its label."""
    holders = lay.mem[lay.mem_ptr[label] : lay.mem_ptr[label + 1]]
    if len(holders) != len(members) or not np.array_equal(np.sort(holders), np.sort(members)):
        _refuse(
            "The multilabel permutation null moves a label among the profiles of its components, so a group must "
            "hold every profile of its label, each with a positive pair."
        )
    labels = [set(lay.lab[lay.lab_ptr[m] : lay.lab_ptr[m + 1]].tolist()) for m in members]
    member_comp = comp[members]
    blocks = []
    lone_base = len(plan.lone_ptr) - 1
    entry_of_member: dict[int, int] = {}
    start = len(plan.entry_profile)
    for c in np.unique(member_comp):
        slots = np.flatnonzero(member_comp == c).tolist()
        others = np.flatnonzero(comp == c)
        others = others[~np.isin(others, members[slots])]
        pool = np.concatenate([members[slots], others])
        for z, s in enumerate(slots):
            entry_of_member[s] = start + z
        start += len(pool)
        entry_lone = [lone_base + s for s in slots] + [-1] * len(others)
        blocks.append((slots, pool, _OWN, [-1 - int(p) for p in pool], [_NO_SET] * len(pool), entry_lone))
    # each member's lone partners: members sharing only this label with it
    for s in range(len(members)):
        partners = [entry_of_member[t] for t in range(len(members)) if t != s and len(labels[s] & labels[t]) == 1]
        plan.lone.extend(partners)
        plan.lone_ptr.append(len(plan.lone))
    return blocks


def _store(lay, result, plan, x, scale, metric, leaf_lo, leaf_n, ops, order, off, n_chunks):
    """Score and sort every (profile, set) the pool entries rank against, once.

    Returns the rows' offsets and keys, and each entry's fixed and subtracted row (``-1`` for none).
    """
    pairs: dict[tuple[int, int], int] = {}

    def rows_of(sets):
        out = np.full(len(plan.entry_profile), -1, np.int64)
        for e, (p, s) in enumerate(zip(plan.entry_profile, sets, strict=True)):
            if s != _NO_SET:
                out[e] = pairs.setdefault((p, s), len(pairs))
        return out

    entry_row, entry_sub_row = rows_of(plan.entry_set), rows_of(plan.entry_sub_set)
    owner = np.fromiter((p for p, _ in pairs), np.int64, len(pairs))
    which = np.fromiter((s for _, s in pairs), np.int64, len(pairs))
    set_ptr = np.concatenate([[0], np.cumsum([len(s) for s in plan.sets])]).astype(np.int64)
    members = np.concatenate(plan.sets) if plan.sets else np.zeros(0, np.int64)
    sizes = np.empty(len(owner), np.int64)
    fixed = which >= 0
    for s in np.unique(which[fixed]):  # a set's size, less its owner when the owner is in it
        rows = np.flatnonzero(which == s)
        sizes[rows] = len(plan.sets[s]) - np.isin(owner[rows], plan.sets[s])
    sizes[~fixed] = result.n_neg[owner[~fixed]]
    store_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    if 4 * int(store_ptr[-1]) > STORE_BUDGET:
        _refuse(
            f"The permutation null would hold {4 * int(store_ptr[-1]) / 2**30:.1f} GiB of sorted similarities, "
            f"above its {STORE_BUDGET // 2**30} GiB budget."
        )
    store = np.empty(int(store_ptr[-1]), np.float32)
    idx = np.flatnonzero(fixed)
    if len(idx):
        _set_rows(
            x, scale, metric, leaf_lo, leaf_n, ops, owner[idx], which[idx], set_ptr, members, store_ptr[idx], store,
            max(1, min(n_chunks, len(idx))),
        )  # fmt: skip
    idx = np.flatnonzero(~fixed)
    if len(idx):
        _own_rows(
            x, scale, metric, leaf_lo, leaf_n, ops, owner[idx], order, off, lay.neg_bucket, lay.neg_diff, lay.lab_ptr,
            lay.lab, store_ptr[idx], store, max(1, min(n_chunks, len(idx))),
        )  # fmt: skip
    return store_ptr, store, entry_row, entry_sub_row
