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
#: A block ranks against its whole ``neg_sameby`` bucket when at most this many of the bucket's profiles are
#: neither its members nor its references; otherwise against its references.
_MAX_EXPLICIT = 64
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


@njit(parallel=True, nogil=True, cache=True)
def _permutation_hits(
    rows, scale, metric, leaf_lo, leaf_n, ops, group_slot, group_block, group_key, thresholds, slot_block, adj_ptr,
    adj, block_slot, block_pool, block_kind, block_extra_ptr, block_extra, entry_profile, entry_row, entry_lone,
    lone_ptr, lone, neg_bucket, neg_diff, lab_ptr, lab, store_ptr, store, items, max_slots, max_pool, max_scored,
    n_chunks,
):  # fmt: skip
    """Draws of each work item ``(group, first draw, end draw)`` whose mAP reaches the group's threshold.

    A group's slots are ``group_slot[g]:group_slot[g + 1]`` and its blocks ``group_block[g]:group_block[g + 1]``;
    block ``b``'s slots are ``block_slot[b]:block_slot[b + 1]`` of the group's slots, in order, and its pool entries
    ``block_pool[b]:block_pool[b + 1]``, its members first. ``entry_row`` is each entry's sorted fixed set.
    """
    one = np.float32(1.0)
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
                    x = entry_profile[e]
                    b = slot_block[s]
                    n_add = 0
                    n_rem = 0
                    kind = block_kind[b]
                    if kind == 0:  # fixed set: the block's references; its undrawn members join, drawn references leave
                        n_mem = block_slot[b + 1] - block_slot[b]
                        for ze in range(block_pool[b], block_pool[b + 1]):
                            if ze == e:
                                continue
                            if ze < block_pool[b] + n_mem:
                                if not drawn[ze - e0]:
                                    added[n_add] = one - _similarity(
                                        rows, scale, x, entry_profile[ze], metric, leaf_lo, leaf_n, ops, stack
                                    )
                                    n_add += 1
                            elif drawn[ze - e0]:
                                removed[n_rem] = one - _similarity(
                                    rows, scale, x, entry_profile[ze], metric, leaf_lo, leaf_n, ops, stack
                                )
                                n_rem += 1
                    elif kind == 1:  # fixed set: the bucket; outsiders and the other drawn profiles leave
                        for t in range(block_extra_ptr[b], block_extra_ptr[b + 1]):
                            removed[n_rem] = one - _similarity(
                                rows, scale, x, block_extra[t], metric, leaf_lo, leaf_n, ops, stack
                            )
                            n_rem += 1
                        for t in range(block_slot[b], block_slot[b + 1]):
                            ze = slot_entry[t - s0]
                            if ze != e:
                                removed[n_rem] = one - _similarity(
                                    rows, scale, x, entry_profile[ze], metric, leaf_lo, leaf_n, ops, stack
                                )
                                n_rem += 1
                    else:  # fixed set: x's own negatives; undrawn lone partners join, drawn negatives leave
                        lx = entry_lone[e]
                        if lx >= 0:
                            for t in range(lone_ptr[lx], lone_ptr[lx + 1]):
                                ze = lone[t]
                                if not drawn[ze - e0]:
                                    added[n_add] = one - _similarity(
                                        rows, scale, x, entry_profile[ze], metric, leaf_lo, leaf_n, ops, stack
                                    )
                                    n_add += 1
                        for t in range(s0, s1):
                            z = entry_profile[slot_entry[t - s0]]
                            if z != x and is_negative(x, z, neg_bucket, neg_diff, True, lab_ptr, lab):
                                removed[n_rem] = one - _similarity(
                                    rows, scale, x, z, metric, leaf_lo, leaf_n, ops, stack
                                )
                                n_rem += 1
                    lo, hi = store_ptr[entry_row[e]], store_ptr[entry_row[e] + 1]
                    deg = 0
                    for t in range(adj_ptr[s], adj_ptr[s + 1]):
                        y = entry_profile[slot_entry[adj[t] - s0]]
                        key = one - _similarity(rows, scale, x, y, metric, leaf_lo, leaf_n, ops, stack)
                        # ties rank the positive first, so only negatives strictly before it count
                        before = _count_below(store, lo, hi, key)
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
        self.block_extra_ptr: list[int] = [0]
        self.block_extra: list[int] = []
        self.entry_profile: list[int] = []
        self.entry_set: list[int] = []
        self.entry_lone: list[int] = []
        self.lone_ptr: list[int] = [0]
        self.lone: list[int] = []
        self.entry_row = np.zeros(0, np.int64)
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

        ``blocks`` lists per block its slots (local), pool entries (members first), kind, extra profiles, and each
        entry's fixed set and lone index.
        """
        s0 = self.group_slot[-1]
        first_block = len(self.block_kind)
        slot_block = np.empty(len(slots_adj), np.int64)
        for b, (slots, pool, kind, extra, entry_set, entry_lone) in enumerate(blocks):
            slot_block[slots] = first_block + b
            self.block_slot.append(self.block_slot[-1] + len(slots))
            self.entry_profile.extend(int(p) for p in pool)
            self.entry_set.extend(entry_set)
            self.entry_lone.extend(entry_lone)
            self.block_pool.append(len(self.entry_profile))
            self.block_kind.append(kind)
            self.block_extra.extend(int(p) for p in extra)
            self.block_extra_ptr.append(len(self.block_extra))
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

    store_ptr, store = _store(lay, result, plan, x, scale, metric, leaf_lo, leaf_n, ops, order, off, n_chunks)
    as64 = lambda a: np.asarray(a, dtype=np.int64)
    items = np.array(
        [(g, d, min(d + _DRAW_CHUNK, null_size)) for g in range(len(groups)) for d in range(0, null_size, _DRAW_CHUNK)],
        dtype=np.int64,
    ).reshape(-1, 3)
    group_slot, group_block, block_pool = as64(plan.group_slot), as64(plan.group_block), as64(plan.block_pool)
    pool_size = block_pool[group_block[1:]] - block_pool[group_block[:-1]]
    max_slots = int(np.diff(group_slot).max(initial=1))
    max_scored = (
        max_slots + int(np.diff(plan.block_extra_ptr).max(initial=0)) + int(np.diff(plan.lone_ptr).max(initial=0))
    )
    max_scored = max(max_scored, int(np.diff(block_pool).max(initial=1)))
    hits = _permutation_hits(
        x, scale, metric, leaf_lo, leaf_n, ops, group_slot, group_block, np.array(plan.group_key, dtype=np.uint64),
        tie_thresholds(np.asarray(scores, dtype=np.float64)), as64(plan.slot_block), as64(plan.adj_ptr), as64(plan.adj),
        as64(plan.block_slot), block_pool, as64(plan.block_kind), as64(plan.block_extra_ptr), as64(plan.block_extra),
        as64(plan.entry_profile), plan.entry_row, as64(plan.entry_lone), as64(plan.lone_ptr), as64(plan.lone),
        lay.neg_bucket, lay.neg_diff, lay.lab_ptr, lay.lab, store_ptr, store, items, max_slots,
        int(pool_size.max(initial=1)), max(max_scored, 1), max(1, min(n_chunks, len(items))),
    )  # fmt: skip
    per_group = np.bincount(items[:, 0], weights=hits, minlength=len(groups))
    return (per_group + 1) / (null_size + 1)


def _single_label_blocks(lay: Layout, plan: _Plan, members: np.ndarray, order: np.ndarray, off: np.ndarray):
    """A group's blocks: members ranked against the same references, each with its pool and fixed sets."""
    signature = [(int(lay.neg_bucket[m]), tuple(lay.neg_diff[m].tolist())) for m in members]
    keys = sorted(set(signature))
    blocks = []
    seen: dict[int, np.ndarray] = {}
    member_set = {int(m) for m in members}
    for key in keys:
        slots = [k for k, s in enumerate(signature) if s == key]
        bucket = order[off[key[0]] : off[key[0] + 1]]
        codes = np.asarray(key[1], dtype=np.int64)
        refs = bucket[(lay.neg_diff[bucket] != codes).all(axis=1)] if len(codes) else bucket
        refs = refs[refs != members[slots[0]]]
        if member_set & set(refs.tolist()):
            _refuse("The permutation null needs each group's references outside it, and a negative joins two members.")
        if key[0] in seen and len(np.intersect1d(seen[key[0]], refs)):
            _refuse("Members of a group rank against overlapping but different references.")
        seen[key[0]] = refs if key[0] not in seen else np.union1d(seen[key[0]], refs)
        block_members = members[slots]
        pool = np.concatenate([block_members, refs])
        outsiders = np.setdiff1d(bucket, pool)
        if len(outsiders) <= _MAX_EXPLICIT:  # rank against the whole bucket, shared by every group in it
            set_id, kind, extra = plan.set_id(bucket), _BUCKET, outsiders
        else:
            set_id, kind, extra = plan.set_id(refs), _REFERENCES, np.zeros(0, np.int64)
        blocks.append((slots, pool, kind, extra, [set_id] * len(pool), [-1] * len(pool)))
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
        blocks.append((slots, pool, _OWN, np.zeros(0, np.int64), [-1 - int(p) for p in pool], entry_lone))
    # each member's lone partners: members sharing only this label with it
    for s in range(len(members)):
        partners = [entry_of_member[t] for t in range(len(members)) if t != s and len(labels[s] & labels[t]) == 1]
        plan.lone.extend(partners)
        plan.lone_ptr.append(len(plan.lone))
    return blocks


def _store(lay, result, plan, x, scale, metric, leaf_lo, leaf_n, ops, order, off, n_chunks):
    """Score and sort every (profile, fixed set) the pool entries rank against, once."""
    pairs = {}
    entry_row = np.empty(len(plan.entry_profile), np.int64)
    for e, (p, s) in enumerate(zip(plan.entry_profile, plan.entry_set, strict=True)):
        entry_row[e] = pairs.setdefault((p, s), len(pairs))
    plan.entry_row = entry_row
    owner = np.array([p for p, _ in pairs], dtype=np.int64)
    which = np.array([s for _, s in pairs], dtype=np.int64)
    set_ptr = np.concatenate([[0], np.cumsum([len(s) for s in plan.sets])]).astype(np.int64)
    members = np.concatenate(plan.sets) if plan.sets else np.zeros(0, np.int64)
    sizes = np.empty(len(owner), np.int64)
    fixed = which >= 0
    if fixed.any():
        within = np.array([np.isin(p, plan.sets[s]) for p, s in zip(owner[fixed], which[fixed], strict=True)])
        sizes[fixed] = np.diff(set_ptr)[which[fixed]] - within
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
    return store_ptr, store
