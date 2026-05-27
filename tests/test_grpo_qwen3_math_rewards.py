# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import importlib.util
from pathlib import Path

import pytest


REWARD_UTILS_PATH = Path(__file__).parents[1] / "learn" / "grpo-qwen3-math" / "reward_utils.py"


@pytest.fixture(scope="module")
def reward_utils():
    spec = importlib.util.spec_from_file_location("grpo_qwen3_math_reward_utils", REWARD_UTILS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"final answer: \boxed{42}", "42"),
        (r"first \boxed{1} then final \boxed{2}", "2"),
        (r"therefore \boxed{\frac{1}{2}}", r"\frac{1}{2}"),
        (r"broken \boxed{\frac{1}{2}", None),
        ("no boxed answer", None),
    ],
)
def test_extract_boxed_handles_last_answer_and_nested_braces(reward_utils, text, expected):
    assert reward_utils.extract_boxed(text) == expected


@pytest.mark.parametrize(
    "answer, expected",
    [
        (r" \dfrac{1}{2}. ", r"\frac{1}{2}"),
        (r"\left( 1, 2 \right)", "(1,2)"),
        (r"\text{meters}", "meters"),
        (r"1,\!234", "1234"),
        (None, None),
    ],
)
def test_normalize_answer_simplifies_common_latex_variants(reward_utils, answer, expected):
    assert reward_utils.normalize_answer(answer) == expected


@pytest.mark.parametrize(
    "prediction, ground_truth",
    [
        (r"\dfrac{1}{2}", r"\frac{1}{2}"),
        (r"1,\!234.", "1234"),
        (r"\text{meters}", "meters"),
    ],
)
def test_answers_equivalent_accepts_normalized_matches(reward_utils, prediction, ground_truth):
    assert reward_utils.answers_equivalent(prediction, ground_truth)


def test_reward_correctness_scores_exact_partial_and_missing_boxed_answers(reward_utils):
    completions = [
        [{"content": r"<think>...</think> \boxed{\dfrac{1}{2}}"}],
        [{"content": r"<think>...</think> \boxed{7}"}],
        [{"content": "I forgot to box it."}],
    ]

    assert reward_utils.reward_correctness(None, completions, [r"\frac{1}{2}", "8", "9"]) == [1.0, 0.1, 0.0]


def test_reward_format_accepts_list_or_string_completions(reward_utils):
    completions = [
        [{"content": "<think>work</think> answer"}],
        "answer without reasoning tags",
    ]

    assert reward_utils.reward_format(None, completions) == [0.5, 0.0]
