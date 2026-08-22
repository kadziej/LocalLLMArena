"""Tests for reasoning-model handling: separating chain-of-thought from the
answer, and the scoring bugs that arise when the two are conflated.

Run with:  python3 -m unittest discover -s tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import split_reasoning, score_case, extract_letter, extract_number


class TestSplitReasoning(unittest.TestCase):
    def test_plain_answer_is_untouched(self):
        self.assertEqual(split_reasoning("The answer is 42."),
                         ("", "The answer is 42."))

    def test_empty_and_none(self):
        self.assertEqual(split_reasoning(""), ("", ""))
        self.assertEqual(split_reasoning(None), ("", ""))

    def test_think_block_is_separated(self):
        r, a = split_reasoning("<think>Let me compute 6*7.</think>\nAnswer: 42")
        self.assertEqual(r, "Let me compute 6*7.")
        self.assertEqual(a, "Answer: 42")

    def test_tag_variants_and_whitespace(self):
        for tag in ("think", "thinking", "reasoning", "reflection", "scratchpad"):
            r, a = split_reasoning(f"< {tag} >hmm</ {tag} >Answer: B")
            self.assertEqual((r, a), ("hmm", "Answer: B"), tag)

    def test_tag_matching_is_case_insensitive(self):
        r, a = split_reasoning("<THINK>hmm</Think>Answer: C")
        self.assertEqual((r, a), ("hmm", "Answer: C"))

    def test_multiple_blocks_are_joined(self):
        r, a = split_reasoning("<think>one</think>partial<think>two</think>done")
        self.assertEqual(r, "one\ntwo")
        self.assertEqual(a, "partialdone")

    def test_unclosed_tag_means_truncated_mid_thought(self):
        # What a max_tokens cut-off during thinking looks like: no answer at all.
        r, a = split_reasoning("<think>step one, step two, step thr")
        self.assertEqual(r, "step one, step two, step thr")
        self.assertEqual(a, "")

    def test_prefilled_opening_tag_leaves_only_a_closing_tag(self):
        # Some chat templates emit "<think>" themselves, so the reply starts
        # mid-thought and we only ever see the closing tag.
        r, a = split_reasoning("weighing the options...</think>\n\nAnswer: D")
        self.assertEqual(r, "weighing the options...")
        self.assertEqual(a, "Answer: D")

    def test_unrelated_markup_is_not_treated_as_reasoning(self):
        text = "Use <thead> in HTML tables."
        self.assertEqual(split_reasoning(text), ("", text))


class TestScoringIgnoresReasoning(unittest.TestCase):
    """Each case here fails if the thinking text is scored instead of the answer."""

    PROMPT = "Which is heaviest?\nA) feather\nB) brick\nC) leaf"

    def test_discarded_candidate_letter_does_not_win(self):
        # The model talks itself out of A while thinking, then answers a bare
        # "B". Letter extraction keys off the last explicit "answer is <X>",
        # which lives in the thinking - so the abandoned guess would win.
        raw = "<think>I believe the answer is A, feathers float.</think>\nB"
        _, answer = split_reasoning(raw)
        self.assertTrue(score_case("multiple_choice", "B", answer, self.PROMPT)[0])
        self.assertFalse(score_case("multiple_choice", "B", raw, self.PROMPT)[0])

    def test_intermediate_number_does_not_win(self):
        # Same shape for numbers: the only "answer" marker is in the thinking
        # and points at a value the model went on to reject.
        raw = "<think>The answer is 41. Wait, recheck.</think>40"
        _, answer = split_reasoning(raw)
        self.assertTrue(score_case("numeric", 40, answer)[0])
        self.assertFalse(score_case("numeric", 40, raw)[0])

    def test_contains_does_not_match_text_only_mentioned_while_thinking(self):
        raw = "<think>Could it be Sydney? No, Canberra is the capital.</think>Canberra"
        _, answer = split_reasoning(raw)
        self.assertFalse(score_case("contains", "Sydney", answer)[0])
        self.assertTrue(score_case("contains", "Sydney", raw)[0])  # false pass

    def test_regex_format_constraint_is_graded_on_the_answer_only(self):
        # "reply in uppercase only" - lowercase thinking must not fail the case.
        raw = "<think>they want shouting, ok</think>HELLO WORLD"
        _, answer = split_reasoning(raw)
        self.assertTrue(score_case("regex_cs", r"^[A-Z\s]+$", answer)[0])
        self.assertFalse(score_case("regex_cs", r"^[A-Z\s]+$", raw)[0])

    def test_exact_match_is_graded_on_the_answer_only(self):
        raw = "<think>capital of Japan</think>Tokyo"
        _, answer = split_reasoning(raw)
        self.assertTrue(score_case("exact_match", "Tokyo", answer)[0])
        self.assertFalse(score_case("exact_match", "Tokyo", raw)[0])


class TestExtractors(unittest.TestCase):
    def test_letter_prefers_last_explicit_statement(self):
        self.assertEqual(extract_letter("First I thought A. Answer: C"), "C")

    def test_letter_is_constrained_to_offered_options(self):
        self.assertEqual(extract_letter("Answer: D", allowed={"A", "B"}), None)

    def test_number_prefers_the_marked_answer(self):
        self.assertEqual(extract_number("half of 48 is 24, so Answer: 24"), 24.0)

    def test_number_handles_thousands_separators_and_currency(self):
        self.assertEqual(extract_number("Answer: $1,250"), 1250.0)

    def test_number_returns_none_when_absent(self):
        self.assertIsNone(extract_number("no digits here"))


if __name__ == "__main__":
    unittest.main()
