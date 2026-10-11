"""research.evolve.space — swing-it pattern_ga_run 의 decode·encode·random_x·메뉴 덮어쓰기 의미
재현."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from krx_quant_core.research.evolve.space import Choice, Ordinal, Real, Space

OFF_ZONE = 0.02
# --- swing-it 정의의 축소 사본(같은 모양: 끔 끝·음수 하한·로그 범위·None·inf·문자열 수준) ---
CONT = (("rs_min", 0.0, 90.0, "lo"), ("sup_c", -1.5, 1.0, None), ("sup_hw", 0.05, 1.5, "hi"),
        ("span_c", math.log(5.0), math.log(120.0), None), ("fb_tol", 0.0, 0.03, None))
NOM = (("stage", (3, 4)), ("entry", ("bottom", "horiz", "trend")),
       ("pyramid", ("off", "1r", "newhigh")))
DISC = (("theta", (0.03, 0.06, 0.08, 0.1)), ("e_min", (None, 0.25, 0.5)),
        ("be_r", (0.5, 1.0, math.inf)),
        ("stop_basis", ("intraday", "intraday")), ("fb_days", (1, 2, 3, 0)))
EVO_GENES = {"trail_start": (0.0,), "exit_safe": (0,), "entry_q": (0,)}
EVO_CONT = {"atr_max": ("atr_max", 0.015, 0.06, "hi")}


def ref_tables(menu):
    """swing-it 모듈 최상단 덮어쓰기 블록 그대로."""
    cont, nom, disc = CONT, NOM, DISC
    if menu:
        def _lv(v):
            return math.inf if v == "inf" else v
        _no, _do, _co = menu.get("nom", {}), menu.get("disc", {}), menu.get("cont", {})

        def _dl(xs):
            t_ = tuple(_lv(x) for x in xs)
            return t_ * 2 if len(t_) == 1 else t_
        disc = (*disc, *((n, _dl(_do[n])) for n in EVO_GENES if n in _do))
        nom = tuple((n, tuple(_lv(x) for x in _no[n]) if n in _no else v) for n, v in nom)
        disc = tuple((n, _dl(_do[n]) if n in _do else v) for n, v in disc)
        def _off(n, c3):                    # e3ccc11: [lo, hi, null] = 끔 없음
            return _co[n][2] if len(_co[n]) >= 3 else c3
        cont = tuple((c[0], float(_co[c[0]][0]), float(_co[c[0]][1]), _off(c[0], c[3]))
                     if c[0] in _co else c for c in cont)
        cont = (*cont, *(EVO_CONT[n][:1] + (float(_co[n][0]), float(_co[n][1]), _off(n, "hi"))
                         for n in EVO_CONT if n in _co))
    return cont, nom, disc


def ref_decode(x, cont, nom, disc):
    n_cont = len(cont) + sum(len(v) for _, v in nom)
    x = np.asarray(x, float)
    out = {}
    for i, (name, lo, hi, off) in enumerate(cont):
        u = float(np.clip(x[i], 0.0, 1.0))
        if (off == "lo" and u <= OFF_ZONE) or (off == "hi" and u >= 1.0 - OFF_ZONE):
            out[name] = None
        else:
            out[name] = round(lo + u * (hi - lo), 4)
    j = len(cont)
    for name, levels in nom:
        v = x[j:j + len(levels)]
        out[name] = levels[int(np.argmax(v))]
        j += len(levels)
    for q, (name, levels) in enumerate(disc):
        k = int(np.clip(round(float(x[n_cont + q])), 0, len(levels) - 1))
        out[name] = levels[k]
    return out


def ref_random_x(rng, n, n_cont, disc_sizes):
    X = rng.random((n, n_cont + len(disc_sizes)))
    for q, k in enumerate(disc_sizes):
        X[:, n_cont + q] = rng.integers(0, k, size=n)
    return X


def core_space():
    genes = [Real(n, lo, hi, off) for n, lo, hi, off in CONT]
    genes.append(Real("atr_max", 0.015, 0.06, "hi", optional=True))
    genes += [Choice(n, lv) for n, lv in NOM]
    genes += [Ordinal(n, lv) for n, lv in DISC]
    genes += [Ordinal(n, (*lv, *lv) if len(lv) == 1 else lv, optional=True)
              for n, lv in EVO_GENES.items()]
    return Space(genes, off_zone=OFF_ZONE, digits=4)


MENUS = [
    None,
    {"nom": {"pyramid": ["off", "1r"]}, "disc": {"theta": [0.06, 0.08, 0.1], "be_r": [1.0, "inf"],
                                                 "exit_safe": [2], "trail_start": [1, 2]},
     "cont": {"sup_c": [-0.15, 0.0], "atr_max": [0.015, 0.06]}},
    {"disc": {"entry_q": [1], "trail_start": [0, 1]}},
    {"cont": {"rs_min": [0.0, 90.0, None], "sup_hw": [0.1, 1.0, None],
              "atr_max": [0.02, 0.05, None],
              "fb_tol": [0.0, 0.03, "lo"]}},          # e3ccc11 G4C 형식: 세 번째 원소 = 끔 끝
]


@pytest.mark.parametrize("mi", range(len(MENUS)))
def test_layout_and_decode_match_reference(mi):
    menu = MENUS[mi]
    cont, nom, disc = ref_tables(menu)
    sp = core_space().override(json.loads(json.dumps(menu)) if menu else None)
    n_cont = len(cont) + sum(len(v) for _, v in nom)
    assert sp.n_cont == n_cont
    assert sp.disc_sizes == tuple(len(v) for _, v in disc)
    assert [g.name for g in sp.reals] == [c[0] for c in cont]
    assert [g.name for g in sp.ordinals] == [d[0] for d in disc]
    rng = np.random.default_rng(mi)
    X = ref_random_x(rng, 3000, sp.n_cont, sp.disc_sizes)
    X[:, :len(cont)] = rng.uniform(-0.2, 1.2, (3000, len(cont)))     # 범위 밖·끔 구역
    X[:500, :len(cont)] = np.round(X[:500, :len(cont)] * 40) / 40     # 끔 경계·동점 부근
    X[:, sp.n_cont:] += rng.uniform(-0.6, 0.6, X[:, sp.n_cont:].shape)  # 정수 축 반올림
    G = sp.decode(X)
    for i in range(len(X)):
        want = ref_decode(X[i], cont, nom, disc)
        got = sp.to_dict(G[i])
        assert set(got) == set(want)
        for k in want:
            a, b = got[k], want[k]
            assert (a is None and b is None) or a == b, (k, a, b)
            if isinstance(b, float):
                assert math.copysign(1.0, a) == math.copysign(1.0, b)   # −0.0 보존


def test_python_round_ties():
    sp = Space([Real("a", 0.0, 1.0)], digits=4)
    xs = (np.arange(0, 10_000) + 0.5) / 1e4
    G = sp.decode(xs[:, None])
    assert all(G[i, 0] == round(float(xs[i]), 4) for i in range(len(xs)))


def test_random_matches_reference():
    sp = core_space().override(MENUS[1])
    a = sp.random(np.random.default_rng([3, 1, 4]), 257)
    b = ref_random_x(np.random.default_rng([3, 1, 4]), 257, sp.n_cont, sp.disc_sizes)
    assert np.array_equal(a, b)


def test_keys_equivalence_classes_match_gkey():
    """같은 swing-it gkey ⇔ 같은 코어 키(굵은 격자로 충돌을 일부러 많이 만든다)."""
    menu = MENUS[1]
    cont, nom, disc = ref_tables(menu)
    sp = core_space().override(menu)
    rng = np.random.default_rng(11)
    X = ref_random_x(rng, 6000, sp.n_cont, sp.disc_sizes)
    X[:, :len(cont)] = np.round(rng.uniform(-0.1, 1.1, (6000, len(cont))) * 4) / 4
    G = sp.decode(X)
    K = sp.keys(G)
    names = [c[0] for c in cont] + [n for n, _ in nom] + [n for n, _ in disc]
    gk = [tuple(ref_decode(x, cont, nom, disc)[k] for k in names) for x in X]
    first: dict = {}
    for k, g in zip(K.tolist(), gk, strict=True):
        first.setdefault(g, k)
        assert first[g] == k
    assert len(set(first.values())) == len(first)          # 다른 gkey → 다른 키


def test_duplicate_levels_fold_and_none_code():
    sp = core_space().override({"disc": {"exit_safe": [2]}})
    assert sp.gene("exit_safe").levels == (2, 2)
    G = sp.decode(np.array([np.r_[np.full(sp.n_cont, 0.5), np.zeros(len(sp.disc_sizes))],
                            np.r_[np.full(sp.n_cont, 0.5), np.ones(len(sp.disc_sizes))]]))
    c = sp.col["exit_safe"]
    assert G[0, c] == G[1, c] == 0
    assert sp.code("e_min", None) == -1
    assert sp.to_dict(G[0])["e_min"] is None


def test_neg_zero_key_merges_but_row_keeps_sign():
    sp = Space([Real("a", -1.0, 1.0)], digits=4)
    G = sp.decode(np.array([[0.49999999], [0.5]]))
    assert math.copysign(1.0, G[0, 0]) == -1.0 and G[1, 0] == 0.0
    K = sp.keys(G)
    assert K[0] == K[1]


def test_override_errors_and_extra():
    sp = core_space()
    with pytest.raises(KeyError):
        sp.override({"disc": {"nope": [1, 2]}})
    with pytest.raises(KeyError):
        sp.override({"cont": {"theta": [0, 1]}})                # 종류 불일치
    s2 = Space([Choice("pyramid", ("1r", "newhigh"), extra=("off",))])
    assert s2.code("pyramid", "off") == 2
    with pytest.raises(KeyError):
        s2.code("pyramid", "upper")


def test_encode_roundtrip_and_row():
    sp = core_space().override(MENUS[1])
    rng = np.random.default_rng(2)
    G = sp.decode(sp.random(rng, 200))
    for r in G:
        g = sp.to_dict(r)
        r2 = sp.row(g)
        assert np.array_equal(r2, r, equal_nan=True)
        assert sp.keys(sp.decode(sp.encode(g)))[0] == sp.keys(r)[0]


def test_bounds_steps_shape():
    sp = core_space()
    assert sp.bounds.shape == (sp.dim, 2) and sp.steps.shape == (sp.dim,)
    assert np.all(sp.bounds[sp.n_cont:, 1] == np.array(sp.disc_sizes) - 1)
