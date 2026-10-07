from __future__ import annotations


METHOD_DISPLAY_ORDER = (
    "m4-boltz-gp_rbf",
    "m4-boltz-iid",
    "birl-boltz-gp_rbf",
    "birl-boltz-iid",
    "pbo-start_end_logistic-gp_rbf",
    "pbo-start_end_logistic-iid",
    "pbo-transition_logistic-gp_rbf",
    "pbo-transition_logistic-iid",
)

METHOD_DISPLAY_LABELS = {
    "m4-boltz-gp_rbf": "Endpoint GP",
    "m4-boltz-iid": "Endpoint i.i.d.",
    "birl-boltz-gp_rbf": "BIRL GP",
    "birl-boltz-iid": "BIRL i.i.d.",
    "pbo-start_end_logistic-gp_rbf": "PBO start-end GP",
    "pbo-start_end_logistic-iid": "PBO start-end i.i.d.",
    "pbo-transition_logistic-gp_rbf": "PBO transition GP",
    "pbo-transition_logistic-iid": "PBO transition i.i.d.",
}

METHOD_COLORS = {
    "m4-boltz-gp_rbf": "#E15759",
    "m4-boltz-iid": "#F28E2B",
    "birl-boltz-gp_rbf": "#4E79A7",
    "birl-boltz-iid": "#76B7B2",
    "pbo-start_end_logistic-gp_rbf": "#59A14F",
    "pbo-start_end_logistic-iid": "#B6992D",
    "pbo-transition_logistic-gp_rbf": "#B07AA1",
    "pbo-transition_logistic-iid": "#9C755F",
}

_METHOD_ORDER_RANK = {label: i for i, label in enumerate(METHOD_DISPLAY_ORDER)}
_DISPLAY_TO_INTERNAL = {
    display_label: internal_label
    for internal_label, display_label in METHOD_DISPLAY_LABELS.items()
}


def method_display_label(label) -> str:
    label_s = str(label)
    return METHOD_DISPLAY_LABELS.get(label_s, label_s)


def method_color(label, fallback: str) -> str:
    label_s = str(label)
    internal_label = _DISPLAY_TO_INTERNAL.get(label_s, label_s)
    return METHOD_COLORS.get(internal_label, fallback)


def method_sort_key(label) -> tuple[int, str]:
    label_s = str(label)
    internal_label = _DISPLAY_TO_INTERNAL.get(label_s, label_s)
    rank = _METHOD_ORDER_RANK.get(internal_label, len(_METHOD_ORDER_RANK))
    return rank, method_display_label(label_s).lower()
