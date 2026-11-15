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
takes out of that set are scored during the draw. The keys between the drawn profiles, which give both a slot's
positives and the profiles it takes out, are scored once per draw. Draw ``j`` of a group is a pure function of the
seed, the group's members and ``j``, so it does not depend on the threads or on the other groups.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from mantispy._core._average_precision import (
    BUILTIN,
    PANEL,
    QUERIES,
    APResult,
    Layout,
    _similarity,
    fill_panel,
    is_negative,
    panel_similarity,
    prepare_rows,
    stack_depth,
    summation_plan,
)
from mantispy._core._map_null import mix, tie_thresholds, uniform
from mantispy._core._numba import N_CHUNKS, group_offsets

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
_TABLE_BUDGET = 2**32
#: A pool entry without a subtracted set.
_NO_SET = -(2**62)
#: Draws per work item of a group of up to 16 slots: enough to amortize a group's setup, few enough to spread one group
#: over the threads. Larger groups take fewer draws per item.
_DRAW_CHUNK = 256
#: Fewest drawn slots whose keys are scored as panels rather than one pair at a time.
_PANEL_SLOTS = 16
#: How a slot's fixed set relates to its negatives.
_REFERENCES, _BUCKET, _OWN = 0, 1, 2


@njit(parallel=True, nogil=True, cache=True)
def _set_rows(rows, scale, metric, leaf_lo, leaf_n, ops, depth, owner, self_at, members, out_ptr, out, items, n_chunks):
    """Keys ``float32(1) - similarity`` of profile ``owner[r]`` to the members of its set, itself left out.

    ``items[k] = (r_lo, r_hi, m_lo, m_hi, set_lo)``: the rows ``r_lo:r_hi`` rank against one set, which starts at
    ``members[set_lo]``, and ``members[m_lo:m_hi]`` is a panel of it. ``self_at[r]`` is the owner's position in its
    set, or ``-1``. The keys land unsorted.
    """
    one = np.float32(1.0)
    for chunk in prange(n_chunks):
        panel = np.zeros((rows.shape[1], PANEL), np.float32)
        acc = np.zeros((QUERIES, 8, PANEL), np.float32)
        stack = np.zeros((depth, QUERIES, PANEL), np.float32)
        sims = np.zeros((QUERIES, PANEL), np.float32)
        qs = np.empty(QUERIES, np.int64)
        for k in range(chunk, items.shape[0], n_chunks):
            r_lo, r_hi, m_lo, m_hi, set_lo = items[k, 0], items[k, 1], items[k, 2], items[k, 3], items[k, 4]
            cand = members[m_lo:m_hi]
            fill_panel(rows, cand, m_hi - m_lo, panel)
            for q0 in range(r_lo, r_hi, QUERIES):
                nq = min(QUERIES, r_hi - q0)
                for q in range(nq):
                    qs[q] = owner[q0 + q]
                panel_similarity(
                    rows, scale, metric, leaf_lo, leaf_n, ops, qs, nq, cand, m_hi - m_lo, panel, acc, stack, sims
                )
                for q in range(nq):
                    r = q0 + q
                    at = self_at[r]
                    for w in range(m_hi - m_lo):
                        t = m_lo - set_lo + w
                        if t != at:
                            out[out_ptr[r] + t - (1 if 0 <= at < t else 0)] = one - sims[q, w]


@njit(parallel=True, nogil=True, cache=True)
def _sort_rows(starts, ends, out, n_chunks):
    """Sort each row ``out[starts[r]:ends[r]]`` in place."""
    for chunk in prange(n_chunks):
        for r in range(chunk, starts.shape[0], n_chunks):
            out[starts[r] : ends[r]].sort()


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
def _gallop(keys, lo, hi, key):
    """The first position in ``[lo, hi)`` whose key is not below ``key``, else ``hi``, for keys sorted ascending.

    The search widens from ``lo`` in doubling steps before it bisects, so a run of rising keys searched in order
    reads each sorted row once from left to right rather than jumping across it per key.
    """
    step = 1
    probe = lo
    while probe < hi and keys[probe] < key:
        lo = probe + 1
        probe = lo + step
        step *= 2
    b = min(probe, hi)
    while lo < b:
        mid = (lo + b) >> 1
        if keys[mid] < key:
            lo = mid + 1
        else:
            b = mid
    return lo


@njit(nogil=True, cache=True, inline="always")
def _key(e1, e2, e0, tidx, table_at, size, n_shared, table, rows, scale, profile, metric, leaf_lo, leaf_n, ops, stack):
    """Rank key between pool entries ``e1`` and ``e2``, from a key table when one holds it, else scored.

    ``tidx[e] >= 0`` is the entry's row in the shared table (the first ``n_shared**2`` keys); ``-1 - i`` makes it row
    ``i`` of its group's private table, which starts at ``table_at`` and holds ``size`` keys per row, one per entry.
    """
    i1 = tidx[e1]
    i2 = tidx[e2]
    if i1 >= 0 and i2 >= 0:
        return table[i1 * n_shared + i2]
    if table_at >= 0:
        if i1 < 0:
            return table[table_at + (-1 - i1) * size + (e2 - e0)]
        return table[table_at + (-1 - i2) * size + (e1 - e0)]
    return np.float32(1.0) - _similarity(rows, scale, profile[e1], profile[e2], metric, leaf_lo, leaf_n, ops, stack)


@njit(parallel=True, nogil=True, cache=True)
def _key_tables(rows, scale, metric, leaf_lo, leaf_n, ops, depth, row_profile, cols, row_at, items, table, n_chunks):
    """Key table rows: ``table[row_at[r] + c]`` is the key of ``row_profile[r]`` to its ``c``-th column.

    ``items[k] = (r_lo, r_hi, c_lo, c_hi, first)``: the rows ``r_lo:r_hi`` share their columns, which start at
    ``cols[first]``, and ``cols[c_lo:c_hi]`` is a panel of them.
    """
    one = np.float32(1.0)
    for chunk in prange(n_chunks):
        panel = np.zeros((rows.shape[1], PANEL), np.float32)
        acc = np.zeros((QUERIES, 8, PANEL), np.float32)
        stack = np.zeros((depth, QUERIES, PANEL), np.float32)
        sims = np.zeros((QUERIES, PANEL), np.float32)
        qs = np.empty(QUERIES, np.int64)
        for k in range(chunk, items.shape[0], n_chunks):
            r_lo, r_hi, c_lo, c_hi, first = items[k, 0], items[k, 1], items[k, 2], items[k, 3], items[k, 4]
            cand = cols[c_lo:c_hi]
            fill_panel(rows, cand, c_hi - c_lo, panel)
            for q0 in range(r_lo, r_hi, QUERIES):
                nq = min(QUERIES, r_hi - q0)
                for q in range(nq):
                    qs[q] = row_profile[q0 + q]
                panel_similarity(
                    rows, scale, metric, leaf_lo, leaf_n, ops, qs, nq, cand, c_hi - c_lo, panel, acc, stack, sims
                )
                for q in range(nq):
                    at = row_at[q0 + q] + c_lo - first
                    for w in range(c_hi - c_lo):
                        table[at + w] = one if cand[w] == qs[q] else one - sims[q, w]


@njit(nogil=True, cache=True)
def _drawn_keys(
    slot_entry, n_slots, e0, tidx, at, size, n_shared, table, rows, scale, profile, metric, leaf_lo, leaf_n, ops,
    stack1, panel, acc, stack, sims, qs, cand, keys,
):  # fmt: skip
    """``keys[a, b]``: the rank key between drawn entries ``a`` and ``b``, from the tables when they hold every pair.

    Otherwise a few slots are scored one pair at a time, and more as panels.
    """
    tabled = at >= 0
    if not tabled and n_shared > 0:
        tabled = True
        for a in range(n_slots):
            if tidx[slot_entry[a]] < 0:
                tabled = False
                break
    if tabled or n_slots < _PANEL_SLOTS:  # a few slots cost less one pair at a time than a panel
        for a in range(n_slots):
            for b in range(a + 1, n_slots):
                v = _key(
                    slot_entry[a], slot_entry[b], e0, tidx, at, size, n_shared, table, rows, scale, profile, metric,
                    leaf_lo, leaf_n, ops, stack1,
                )  # fmt: skip
                keys[a, b] = v
                keys[b, a] = v
        return
    one = np.float32(1.0)
    for c0 in range(0, n_slots, PANEL):  # the upper triangle, a panel at a time
        nc = min(PANEL, n_slots - c0)
        for w in range(nc):
            cand[w] = profile[slot_entry[c0 + w]]
        fill_panel(rows, cand, nc, panel)
        for q0 in range(0, c0 + nc - 1, QUERIES):
            nq = min(QUERIES, c0 + nc - 1 - q0)
            for q in range(nq):
                qs[q] = profile[slot_entry[q0 + q]]
            panel_similarity(rows, scale, metric, leaf_lo, leaf_n, ops, qs, nq, cand, nc, panel, acc, stack, sims)
            for q in range(nq):
                a = q0 + q
                for w in range(max(0, a + 1 - c0), nc):
                    v = one - sims[q, w]
                    keys[a, c0 + w] = v
                    keys[c0 + w, a] = v


@njit(nogil=True, cache=True, inline="always")
def _sort_head(values, n):
    values[:n].sort()


@njit(parallel=True, nogil=True, cache=True)
def _permutation_hits(
    rows, scale, metric, leaf_lo, leaf_n, ops, depth, group_slot, group_block, group_key, thresholds, group_table,
    group_size, tidx, n_shared, table, slot_block, adj_ptr, adj, block_slot, block_pool, block_kind, profile,
    entry_row, entry_sub_row, entry_lone, lone_ptr, lone, neg_bucket, neg_diff, lab_ptr, lab, store_ptr, store, items,
    max_slots, max_pool, max_scored, n_chunks,
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
        stack1 = np.empty(64, np.float32)
        panel = np.zeros((rows.shape[1], PANEL), np.float32)
        acc = np.zeros((QUERIES, 8, PANEL), np.float32)
        stack = np.zeros((depth, QUERIES, PANEL), np.float32)
        sims = np.zeros((QUERIES, PANEL), np.float32)
        qs = np.empty(QUERIES, np.int64)
        cand = np.empty(PANEL, np.int64)
        keys = np.empty((max_slots, max_slots), np.float32)
        slot_entry = np.empty(max_slots, np.int64)
        drawn = np.zeros(max_pool, np.bool_)
        chosen = np.empty(max_slots, np.int64)
        added = np.empty(max_scored, np.float32)
        removed = np.empty(max_scored, np.float32)
        pos_key = np.empty(max_slots, np.float32)
        for k in range(chunk, items.shape[0], n_chunks):
            g, d_lo, d_hi = items[k, 0], items[k, 1], items[k, 2]
            s0, s1 = group_slot[g], group_slot[g + 1]
            b0, b1 = group_block[g], group_block[g + 1]
            e0 = block_pool[b0]
            at = group_table[g]
            size = group_size[g]
            for j in range(d_lo, d_hi):
                state = mix(group_key[g] ^ mix(np.uint64(j) + np.uint64(0x632BE59BD9B4E019)))
                for b in range(b0, b1):  # Floyd's k-subset of the block's pool, then a uniform order
                    kk = block_slot[b + 1] - block_slot[b]
                    pp = block_pool[b + 1] - block_pool[b]
                    for c in range(kk):
                        t = pp - kk + c
                        state, u = uniform(state)
                        r = min(np.int64(u * (t + 1)), t)
                        for z in range(c):
                            if chosen[z] == r:
                                r = t
                                break
                        chosen[c] = r
                    for z in range(kk - 1, 0, -1):
                        state, u = uniform(state)
                        r = min(np.int64(u * (z + 1)), z)
                        chosen[z], chosen[r] = chosen[r], chosen[z]
                    for c in range(kk):
                        e = block_pool[b] + chosen[c]
                        slot_entry[block_slot[b] + c - s0] = e
                        drawn[e - e0] = True
                _drawn_keys(
                    slot_entry, s1 - s0, e0, tidx, at, size, n_shared, table, rows, scale, profile, metric, leaf_lo,
                    leaf_n, ops, stack1, panel, acc, stack, sims, qs, cand, keys,
                )  # fmt: skip
                total = 0.0
                for s in range(s0, s1):
                    a = s - s0
                    e = slot_entry[a]
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
                                        e, ze, e0, tidx, at, size, n_shared, table, rows, scale, profile, metric,
                                        leaf_lo, leaf_n, ops, stack1,
                                    )  # fmt: skip
                                    n_add += 1
                        for t in range(s1 - s0):
                            if t != a and is_negative(
                                profile[e], profile[slot_entry[t]], neg_bucket, neg_diff, True, lab_ptr, lab
                            ):
                                removed[n_rem] = keys[a, t]
                                n_rem += 1
                    else:
                        n_mem = block_slot[b + 1] - block_slot[b]
                        if kind == 0:  # fixed set: the block's references; its undrawn members join
                            for ze in range(block_pool[b], block_pool[b] + n_mem):
                                if ze != e and not drawn[ze - e0]:
                                    added[n_add] = _key(
                                        e, ze, e0, tidx, at, size, n_shared, table, rows, scale, profile, metric,
                                        leaf_lo, leaf_n, ops, stack1,
                                    )  # fmt: skip
                                    n_add += 1
                        for t in range(block_slot[b] - s0, block_slot[b + 1] - s0):  # the other drawn profiles leave
                            if t != a and (kind == 1 or slot_entry[t] >= block_pool[b] + n_mem):
                                removed[n_rem] = keys[a, t]
                                n_rem += 1
                    _sort_head(added, n_add)
                    _sort_head(removed, n_rem)
                    deg = adj_ptr[s + 1] - adj_ptr[s]
                    for t in range(deg):
                        pos_key[t] = keys[a, adj[adj_ptr[s] + t] - s0]
                    _sort_head(pos_key, deg)
                    lo, hi = store_ptr[entry_row[e]], store_ptr[entry_row[e] + 1]
                    sub = entry_sub_row[e]
                    sub_lo, sub_hi = (store_ptr[sub], store_ptr[sub + 1]) if sub >= 0 else (0, 0)
                    # the positives rise, so each count resumes where the previous one stopped
                    i_row, i_sub, i_add, i_rem = lo, sub_lo, 0, 0
                    acc_ap = 0.0
                    for q in range(deg):
                        key = pos_key[q]
                        # ties rank the positive first, so only negatives strictly before it count
                        i_row = _gallop(store, i_row, hi, key)
                        i_sub = _gallop(store, i_sub, sub_hi, key)
                        i_add = _gallop(added, i_add, n_add, key)
                        i_rem = _gallop(removed, i_rem, n_rem, key)
                        before = (i_row - lo) - (i_sub - sub_lo) + i_add - i_rem
                        acc_ap += (q + 1) / (q + 1 + before)
                    total += acc_ap / deg
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


def _ptr(counts: list[int]) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)


def _cat(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, np.int64)


class _Plan:
    """Flat arrays of every group's slots, blocks and pool entries, and the fixed sets the entries rank against.

    Groups are appended as numpy arrays and joined by :meth:`finish`, which sets the flat arrays as attributes.
    """

    def __init__(self) -> None:
        self.n_slots = 0
        self.n_blocks = 0
        self.n_entries = 0
        self.n_lone = 0
        self.group_key: list[np.uint64] = []
        self._group_slots: list[int] = []
        self._group_blocks: list[int] = []
        self._slot_block: list[np.ndarray] = []
        self._adj_counts: list[np.ndarray] = []
        self._adj: list[np.ndarray] = []
        self._block_slots: list[int] = []
        self._block_pools: list[int] = []
        self._block_kind: list[int] = []
        self._entry_profile: list[np.ndarray] = []
        self._entry_set: list[np.ndarray] = []
        self._entry_sub_set: list[np.ndarray] = []
        self._entry_lone: list[np.ndarray] = []
        self._lone_counts: list[int] = []
        self._lone: list[np.ndarray] = []
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

    def add_lone(self, partners: np.ndarray) -> None:
        """Append one member's lone partners, as pool entries."""
        self._lone.append(np.asarray(partners, dtype=np.int64))
        self._lone_counts.append(len(partners))
        self.n_lone += 1

    def add_group(self, key, slots_adj, blocks):
        """Append one group.

        ``slots_adj[s]`` lists slot ``s``'s positives as local slots. ``blocks`` lists per block its slots (local),
        pool entries (members first), kind, and each entry's fixed set, subtracted set (:data:`_NO_SET` for none) and
        lone index.
        """
        slot_block = np.empty(len(slots_adj), np.int64)
        for b, (slots, pool, kind, entry_set, entry_sub_set, entry_lone) in enumerate(blocks):
            slot_block[slots] = self.n_blocks + b
            self._block_slots.append(len(slots))
            self._block_pools.append(len(pool))
            self._block_kind.append(kind)
            self._entry_profile.append(np.asarray(pool, dtype=np.int64))
            self._entry_set.append(np.asarray(entry_set, dtype=np.int64))
            self._entry_sub_set.append(np.asarray(entry_sub_set, dtype=np.int64))
            self._entry_lone.append(np.asarray(entry_lone, dtype=np.int64))
            self.n_entries += len(pool)
        self.n_blocks += len(blocks)
        # slots are numbered block by block, so a block's slots are contiguous
        order = np.concatenate([np.asarray(slots, dtype=np.int64) for slots, *_ in blocks])
        new_of_old = np.empty(len(order), np.int64)
        new_of_old[order] = np.arange(len(order))
        adjacent = [np.asarray(slots_adj[old], dtype=np.int64) for old in order]
        self._slot_block.append(slot_block[order])
        self._adj_counts.append(np.array([len(a) for a in adjacent], dtype=np.int64))
        self._adj.append(self.n_slots + new_of_old[_cat(adjacent)])
        self.n_slots += len(order)
        self._group_slots.append(len(order))
        self._group_blocks.append(len(blocks))
        self.group_key.append(key)

    def finish(self) -> None:
        """Join the groups into the flat arrays the kernels read."""
        self.group_slot = _ptr(self._group_slots)
        self.group_block = _ptr(self._group_blocks)
        self.slot_block = _cat(self._slot_block)
        self.adj_ptr = _ptr([int(c) for counts in self._adj_counts for c in counts])
        self.adj = _cat(self._adj)
        self.block_slot = _ptr(self._block_slots)
        self.block_pool = _ptr(self._block_pools)
        self.block_kind = np.asarray(self._block_kind, dtype=np.int64)
        self.entry_profile = _cat(self._entry_profile)
        self.entry_set = _cat(self._entry_set)
        self.entry_sub_set = _cat(self._entry_sub_set)
        self.entry_lone = _cat(self._entry_lone)
        self.lone_ptr = _ptr(self._lone_counts)
        self.lone = _cat(self._lone)


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
    depth = stack_depth(ops)
    metric = BUILTIN[distance]
    order, off = group_offsets(lay.neg_bucket, lay.n_buckets)
    n_labels = int(result.row_label.max(initial=0)) + 1
    row_key = result.row_profile * n_labels + result.row_label
    comp = (
        _components(order, off, lay.neg_bucket, lay.neg_diff, lay.lab_ptr, lay.lab)
        if lay.disjoint
        else np.zeros(lay.n, np.int64)
    )

    plan = _Plan()
    local = np.full(lay.n, -1, np.int64)
    for ix in groups:
        ap_rows = rows[ix]
        members = result.row_profile[ap_rows]
        label = int(result.row_label[ap_rows[0]])
        local[members] = np.arange(len(members))
        starts, ends = result.pos_ptr[ap_rows], result.pos_ptr[ap_rows + 1]
        partners = result.partner[_cat([np.arange(a, b) for a, b in zip(starts, ends, strict=True)])]
        # a positive of a multilabel row is a row of the same label
        found = np.isin(partners * n_labels + label, row_key) if lay.disjoint else np.ones(len(partners), bool)
        partner_slots = local[partners]
        if not found.all() or (partner_slots < 0).any():
            _refuse(
                "The permutation null needs each group's positives to be its other members, and a group has "
                "a positive outside it (for example a member dropped for having no negative pair)."
            )
        slots_adj = np.split(partner_slots, np.cumsum(ends - starts)[:-1])
        if lay.disjoint:
            blocks = _label_blocks(lay, plan, members, label, comp)
        else:
            blocks = _single_label_blocks(lay, plan, members, local, order, off)
        plan.add_group(_group_key(seed, members, label if lay.disjoint else -1), slots_adj, blocks)
        local[members] = -1
    plan.finish()

    store_ptr, store, entry_row, entry_sub_row = _store(
        lay, result, plan, x, scale, metric, leaf_lo, leaf_n, ops, depth, order, off, n_chunks
    )
    group_slot, group_block, block_pool = plan.group_slot, plan.group_block, plan.block_pool
    profile = plan.entry_profile
    group_table, group_size, tidx, n_shared, table = _tables(
        plan, group_slot, group_block, block_pool, profile, null_size, x, scale, metric, leaf_lo, leaf_n, ops, depth,
        n_chunks,
    )  # fmt: skip
    items = _draw_items(np.diff(group_slot), null_size)
    pool_size = block_pool[group_block[1:]] - block_pool[group_block[:-1]]
    max_slots = int(np.diff(group_slot).max(initial=1))
    max_scored = 2 * max_slots + int(np.diff(plan.lone_ptr).max(initial=0)) + 1
    hits = _permutation_hits(
        x, scale, metric, leaf_lo, leaf_n, ops, depth, group_slot, group_block, np.array(plan.group_key, np.uint64),
        tie_thresholds(np.asarray(scores, dtype=np.float64)), group_table, group_size, tidx, n_shared, table,
        plan.slot_block, plan.adj_ptr, plan.adj, plan.block_slot, block_pool, plan.block_kind, profile, entry_row,
        entry_sub_row, plan.entry_lone, plan.lone_ptr, plan.lone, lay.neg_bucket, lay.neg_diff, lay.lab_ptr, lay.lab,
        store_ptr, store, items, max_slots, int(pool_size.max(initial=1)), max_scored, max(1, min(n_chunks, len(items))),
    )  # fmt: skip
    per_group = np.bincount(items[:, 0], weights=hits, minlength=len(groups))
    return (per_group + 1) / (null_size + 1)


def _draw_items(n_slots: np.ndarray, null_size: int) -> np.ndarray:
    """Work items ``(group, first draw, end draw)``, the costliest first.

    A draw costs about the square of its group's slots, for the keys between them, so a large group's draws are
    spread over more items, which keeps one group from holding a thread after the others have finished.
    """
    per_item = np.clip(_DRAW_CHUNK * 16**2 // np.maximum(n_slots, 1) ** 2, 1, _DRAW_CHUNK)
    n_items = -(-null_size // per_item)
    group = np.repeat(np.arange(len(n_slots)), n_items)
    first = (np.arange(len(group)) - np.repeat(np.cumsum(n_items) - n_items, n_items)) * per_item[group]
    items = np.stack([group, first, np.minimum(first + per_item[group], null_size)], axis=1).astype(np.int64)
    return items[np.argsort(-n_slots[group], kind="stable")]


def _panel_items(row_lo: int, row_hi: int, col_lo: int, col_hi: int, first: int) -> list[tuple[int, ...]]:
    """Work items of rows ``row_lo:row_hi`` that share the columns ``col_lo:col_hi``, a panel of columns at a time."""
    return [
        (r, min(r + PANEL, row_hi), c, min(c + PANEL, col_hi), first)
        for r in range(row_lo, row_hi, PANEL)
        for c in range(col_lo, col_hi, PANEL)
    ]


def _tables(plan, group_slot, group_block, block_pool, profile, null_size, x, scale, metric, leaf_lo, leaf_n, ops,
            depth, n_chunks):  # fmt: skip
    """Key tables, wherever a draw would otherwise score more pairs than a table holds.

    Profiles in the pools of several groups (a plate's controls, or whole plates) share one table; each group's own
    profiles (its members) get a table to the rest of its pool. Returns each group's private table offset (``-1``
    for none) and width, each entry's table row (:func:`_key`), the shared table's width and the tables.
    """
    n_groups = len(group_slot) - 1
    adj_ptr = plan.adj_ptr
    e0 = block_pool[group_block[:-1]]
    size = (block_pool[group_block[1:]] - e0).astype(np.int64)
    n_slots = np.diff(group_slot)
    scored = null_size * (adj_ptr[group_slot[1:]] - adj_ptr[group_slot[:-1]] + 2 * n_slots * n_slots)
    owner = np.repeat(np.arange(n_groups), size)
    # profiles in the pools of more than one group
    pairs = np.unique(profile * max(n_groups, 1) + owner)
    candidates, n_owners = np.unique(pairs // max(n_groups, 1), return_counts=True)
    shared = candidates[n_owners > 1]
    if not (4 * len(shared) ** 2 <= _TABLE_BUDGET and len(shared) ** 2 < np.minimum(size * size, scored).sum()):
        shared = shared[:0]
    n_shared = len(shared)
    tidx = np.searchsorted(shared, profile).astype(np.int64)
    is_shared = (
        (tidx < n_shared) & (shared[np.minimum(tidx, max(n_shared - 1, 0))] == profile)
        if n_shared
        else (np.zeros(len(profile), bool))
    )
    private = ~is_shared
    # private entries count from -1 down within their group
    rank = np.cumsum(private) - 1
    first = np.concatenate([[0], np.cumsum(private)])[e0]
    tidx = np.where(is_shared, tidx, -1 - (rank - np.repeat(first, size)))
    group_table = np.full(n_groups, -1, np.int64)
    used = n_shared * n_shared
    cols = np.concatenate([shared, profile]).astype(np.int64)
    row_owner = [shared]
    row_at = [np.arange(n_shared, dtype=np.int64) * n_shared]
    items = _panel_items(0, n_shared, 0, n_shared, 0)
    n_private = np.bincount(owner[private], minlength=n_groups)
    cost = n_private * size
    for g in np.flatnonzero((cost <= scored) & (cost > 0)).tolist():  # in group order while the budget lasts
        if 4 * (used + int(cost[g])) <= _TABLE_BUDGET:
            group_table[g] = used
            used += int(cost[g])
    tabled = group_table >= 0
    # a tabled group's rows are its private entries, in entry order; its columns its pool, after the shared profiles
    rows_e = np.flatnonzero(private & tabled[owner])
    row_owner.append(profile[rows_e])
    row_at.append(group_table[owner[rows_e]] + (-1 - tidx[rows_e]) * size[owner[rows_e]])
    g_rows = np.flatnonzero(tabled)
    n_g, size_g = n_private[g_rows], size[g_rows]
    row0 = n_shared + np.concatenate([[0], np.cumsum(n_g)[:-1]]).astype(np.int64)
    col0 = e0[g_rows] + n_shared
    n_rb, n_cb = -(-n_g // PANEL), -(-size_g // PANEL)
    which = np.repeat(np.arange(len(g_rows)), n_rb * n_cb)
    within = np.arange(len(which)) - np.repeat(np.cumsum(n_rb * n_cb) - n_rb * n_cb, n_rb * n_cb)
    rb, cb = within // n_cb[which], within % n_cb[which]
    r_lo, c_lo = row0[which] + rb * PANEL, col0[which] + cb * PANEL
    private_items = np.stack(
        [
            r_lo,
            np.minimum(r_lo + PANEL, row0[which] + n_g[which]),
            c_lo,
            np.minimum(c_lo + PANEL, col0[which] + size_g[which]),
            col0[which],
        ],
        axis=1,
    )
    work = np.concatenate([np.array(items, dtype=np.int64).reshape(-1, 5), private_items]).astype(np.int64)
    table = np.empty(used, np.float32)
    if len(work):
        _key_tables(
            x, scale, metric, leaf_lo, leaf_n, ops, depth, _cat(row_owner), cols, _cat(row_at), work, table,
            max(1, min(n_chunks, len(work))),
        )  # fmt: skip
    return group_table, size, tidx, n_shared, table


def _single_label_blocks(
    lay: Layout, plan: _Plan, members: np.ndarray, local: np.ndarray, order: np.ndarray, off: np.ndarray
):
    """A group's blocks: members ranked against the same references, each with its pool and fixed sets.

    ``local`` marks the group's members (``>= 0``).
    """
    signature = [(int(lay.neg_bucket[m]), tuple(lay.neg_diff[m].tolist())) for m in members]
    blocks = []
    seen: dict[int, np.ndarray] = {}
    for key in sorted(set(signature)):
        slots = [k for k, s in enumerate(signature) if s == key]
        bucket = order[off[key[0]] : off[key[0] + 1]]
        codes = np.asarray(key[1], dtype=np.int64)
        refs = bucket[(lay.neg_diff[bucket] != codes).all(axis=1)] if len(codes) else bucket
        if (local[refs] >= 0).any():
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
        n = len(pool)
        blocks.append((slots, pool, kind, np.full(n, set_id), np.full(n, sub_id), np.full(n, -1)))
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
    lone_base = plan.n_lone
    entry_of_member: dict[int, int] = {}
    start = plan.n_entries
    for c in np.unique(member_comp):
        slots = np.flatnonzero(member_comp == c).tolist()
        others = np.flatnonzero(comp == c)
        others = others[~np.isin(others, members[slots])]
        pool = np.concatenate([members[slots], others])
        for z, s in enumerate(slots):
            entry_of_member[s] = start + z
        start += len(pool)
        entry_lone = np.concatenate([lone_base + np.asarray(slots, dtype=np.int64), np.full(len(others), -1)])
        blocks.append((slots, pool, _OWN, -1 - pool, np.full(len(pool), _NO_SET), entry_lone))
    # each member's lone partners: members sharing only this label with it
    for s in range(len(members)):
        plan.add_lone(
            np.array(
                [entry_of_member[t] for t in range(len(members)) if t != s and len(labels[s] & labels[t]) == 1],
                dtype=np.int64,
            )
        )
    return blocks


def _store(lay, result, plan, x, scale, metric, leaf_lo, leaf_n, ops, depth, order, off, n_chunks):
    """Score and sort every (profile, set) the pool entries rank against, once.

    Returns the rows' offsets and keys, and each entry's fixed and subtracted row (``-1`` for none).
    """
    n = lay.n
    # a (profile, set) pair as one integer; a set id below zero is a profile's own negatives
    width = len(plan.sets) + n + 1
    has_sub = plan.entry_sub_set != _NO_SET
    codes = np.concatenate(
        [
            plan.entry_profile * width + plan.entry_set + n + 1,
            plan.entry_profile[has_sub] * width + plan.entry_sub_set[has_sub] + n + 1,
        ]
    )
    pairs, inverse = np.unique(codes, return_inverse=True)
    entry_row = inverse[: len(plan.entry_profile)].astype(np.int64)
    entry_sub_row = np.full(len(plan.entry_profile), -1, np.int64)
    entry_sub_row[has_sub] = inverse[len(plan.entry_profile) :]
    owner = pairs // width
    which = pairs % width - n - 1
    set_ptr = _ptr([len(s) for s in plan.sets])
    members = _cat(plan.sets)
    fixed = which >= 0
    # a fixed set's row leaves out its owner when the owner is in it
    self_at = np.full(len(owner), -1, np.int64)
    rows_f = np.flatnonzero(fixed)
    lo_f = set_ptr[which[rows_f]]
    hi_f = set_ptr[which[rows_f] + 1]
    found = np.array(
        [np.searchsorted(members[a:b], p) for a, b, p in zip(lo_f, hi_f, owner[rows_f], strict=True)], dtype=np.int64
    ).reshape(-1)
    inside = (found < hi_f - lo_f) & (members[np.minimum(lo_f + found, max(len(members) - 1, 0))] == owner[rows_f])
    self_at[rows_f[inside]] = found[inside]
    sizes = np.empty(len(owner), np.int64)
    sizes[rows_f] = hi_f - lo_f - inside
    sizes[~fixed] = result.n_neg[owner[~fixed]]
    store_ptr = _ptr(sizes)
    if 4 * int(store_ptr[-1]) > STORE_BUDGET:
        _refuse(
            f"The permutation null would hold {4 * int(store_ptr[-1]) / 2**30:.1f} GiB of sorted similarities, "
            f"above its {STORE_BUDGET // 2**30} GiB budget."
        )
    store = np.empty(int(store_ptr[-1]), np.float32)
    if len(rows_f):
        # rows of one set are scored together against it, a panel of its members at a time
        by_set = rows_f[np.argsort(which[rows_f], kind="stable")]
        bounds = np.flatnonzero(np.diff(which[by_set], prepend=-1, append=-2))
        items = []
        for a, b in zip(bounds[:-1], bounds[1:], strict=True):
            s = which[by_set[a]]
            items += _panel_items(a, b, int(set_ptr[s]), int(set_ptr[s + 1]), int(set_ptr[s]))
        work = np.array(items, dtype=np.int64).reshape(-1, 5)
        _set_rows(
            x, scale, metric, leaf_lo, leaf_n, ops, depth, owner[by_set], self_at[by_set], members,
            store_ptr[by_set], store, work, max(1, min(n_chunks, len(work))),
        )  # fmt: skip
        _sort_rows(store_ptr[rows_f], store_ptr[rows_f + 1], store, max(1, min(n_chunks, len(rows_f))))
    idx = np.flatnonzero(~fixed)
    if len(idx):
        _own_rows(
            x, scale, metric, leaf_lo, leaf_n, ops, owner[idx], order, off, lay.neg_bucket, lay.neg_diff, lay.lab_ptr,
            lay.lab, store_ptr[idx], store, max(1, min(n_chunks, len(idx))),
        )  # fmt: skip
    return store_ptr, store, entry_row, entry_sub_row
