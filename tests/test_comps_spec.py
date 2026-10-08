from models import comps_spec as cs


def test_units_features_and_markets_are_consistent():
    assert set(cs.FEATURES) == set(cs.UNITS)
    assert all(u in cs.UNITS for units in cs.MARKET_UNITS.values() for u in units)
    for unit, feats in cs.FEATURES.items():
        names = [f.name for f in feats]
        assert len(names) == len(set(names)), unit


def test_spaces_match_sources():
    for feats in cs.FEATURES.values():
        for f in feats:
            if f.space == "base":
                assert f.first_season == cs.BASE_START and not any(s in f.source for s in ("ftn", "pfr", "participation"))
            else:
                assert f.first_season >= cs.PFR_START
    assert all(f.space == "extended" for f in cs.FEATURES["coverage_mix"])


def test_team_level_markets_have_no_player_archetypes():
    for m in ("total", "spread", "moneyline"):
        assert not set(cs.MARKET_UNITS[m]) & set(cs.ARCHETYPE_UNITS)


def test_constants():
    assert (cs.SIM_THRESHOLD, cs.MIN_NEFF, cs.SHRINK_K, cs.TEAM_CAP_PER_SEARCH, cs.TEAM_CAP_AVG_ACROSS_SEARCHES) == (0.70, 2, 5, 0.25, 0.15)
    assert cs.QUALITY == {"observed": 1.0, "derived": 0.9, "estimated": 0.65}
