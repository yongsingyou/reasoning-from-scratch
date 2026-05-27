import re

try:
    from math_verify import LatexExtractionConfig, parse, verify
except ImportError:
    LatexExtractionConfig = None
    parse = None
    verify = None


def extract_boxed(text):
    answers = []
    start = 0
    marker = r"\boxed{"
    while True:
        box_start = text.find(marker, start)
        if box_start == -1:
            break

        content_start = box_start + len(marker)
        depth = 1
        i = content_start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1

        if depth == 0:
            answers.append(text[content_start:i - 1].strip())
            start = i
        else:
            break

    return answers[-1] if answers else None


def normalize_answer(text):
    if text is None:
        return None

    text = str(text).strip()
    text = text.strip("$")
    text = re.sub(r"\\(?:left|right|displaystyle|textstyle|scriptstyle|scriptscriptstyle)", "", text)
    text = text.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = text.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "").replace(r"\:", "")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return text


def answers_equivalent(pred, gt):
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if pred_norm == gt_norm:
        return True

    if parse is None or verify is None or LatexExtractionConfig is None:
        return False

    try:
        parsed_pred = parse(pred, extraction_config=[LatexExtractionConfig()])
        parsed_gt = parse(gt, extraction_config=[LatexExtractionConfig()])
        return bool(parsed_pred and parsed_gt and verify(parsed_pred, parsed_gt))
    except Exception:
        return False


def reward_correctness(prompts, completions, answer, **kwargs):
    scores = []
    for completion, gt in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        extracted = extract_boxed(text)
        if extracted is None:
            scores.append(0.0)
        elif answers_equivalent(extracted, gt):
            scores.append(1.0)
        else:
            scores.append(0.1)
    return scores


def reward_format(prompts, completions, **kwargs):
    scores = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        has_think = bool(re.search(r"<think>.*?</think>", text, re.DOTALL))
        scores.append(0.5 if has_think else 0.0)
    return scores
