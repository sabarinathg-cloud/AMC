"""Spoken digit strings must be maskable; ordinary speech about numbers must not be.

The positive cases are transcripts taken verbatim from 2025 shard-0 that shipped with
no mask. The negative cases are the over-masking risk: this corpus is full of clinical
readings and dates, and bleeping those would destroy analytic value for no privacy gain.
"""
from __future__ import annotations

import unittest

from amc_pipeline.spoken_numbers import find_spoken_number_runs


def types(text: str) -> list[str]:
    return [r.entity_type for r in find_spoken_number_runs(text)]


class LeakedRealTranscriptsTest(unittest.TestCase):
    """Every one of these shipped unmasked from the live run."""

    CASES = [
        "Please leave your message for four zero eight eight nine nine one six nine eight.",
        "Hi, you have reached six eight eight zero five four four. Leave me a message, "
        "and I'll get back to you as soon as I can.",
        "Four, zero, eight, four, one, five, seven, one, eight, seven is not available.",
        "Two, three, one, seven, seven, seven, two, four, one, seven is not available.",
        "Forwarded to an automatic voice message system. Five seven zero nine five four "
        "two one one four is not available.",
        "Please give us a call back at your earliest convenience at three one three five "
        "two four zero zero seven five",
    ]

    def test_each_is_detected(self):
        for text in self.CASES:
            with self.subTest(text=text[:48]):
                self.assertIn("PHONE", types(text))

    def test_span_covers_the_digits(self):
        text = "Please leave your message for four zero eight eight nine nine one six nine eight."
        run = find_spoken_number_runs(text)[0]
        self.assertEqual(run.digits, 10)
        # The span has to start at the first digit word and end at the last, or masking
        # would either clip the number or eat the surrounding speech.
        self.assertTrue(text[run.start:].startswith("four zero"))
        self.assertTrue(text[: run.end].endswith("nine eight"))
        self.assertNotIn("message", run.text)


class OrdinarySpeechIsLeftAloneTest(unittest.TestCase):
    CASES = [
        "Blood pressure was one twenty over eighty and the pulse was seventy two.",
        "I've been on it for about twenty five years, maybe thirty.",
        "It comes in that big box at Sam's Club, so that's how I buy it.",
        "We're not available on weekends or holidays.",
        "Take one tablet twice a day, two in the morning and one at night.",
        "Yeah, no, it's fine. Okay. Mm-hmm.",
        "A hundred and twenty thousand people are in the program.",
        "One, two, three, testing.",
        "Monday through Friday, nine to four thirty.",
        "I had my coffee, so we're just calling since you're in the program.",
    ]

    def test_no_spans(self):
        for text in self.CASES:
            with self.subTest(text=text[:48]):
                self.assertEqual(types(text), [], f"over-masked: {text}")


class CuedShortRunsTest(unittest.TestCase):
    def test_member_id_is_detected(self):
        self.assertIn("ID", types("Your member ID is four four four two."))

    def test_date_of_birth_is_detected(self):
        self.assertIn("ID", types("And your date of birth, one two nineteen fifty."))

    def test_short_run_without_a_cue_is_ignored(self):
        # Four digits with no identifier cue is not enough to mask on.
        self.assertEqual(types("I'll take four, five, six and seven."), [])

    def test_cue_plus_ordinary_quantity_is_not_masked(self):
        # A cue nearby must not turn plain speech about quantities into PII.
        self.assertEqual(types("What's your number? About twenty or thirty a day."), [])
        self.assertEqual(types("Your member benefits cover a hundred and twenty visits."), [])

    def test_cue_must_be_nearby(self):
        far = ("Your member ID, well, let me find it, I think I wrote it down somewhere "
               "in the kitchen, hold on a moment, okay, four four four two.")
        self.assertEqual(types(far), [])


class RunBoundaryTest(unittest.TestCase):
    def test_connector_words_do_not_split_a_number(self):
        text = "at area code four one five, five five five, one two one two"
        self.assertIn("PHONE", types(text))

    def test_real_words_do_split_a_number(self):
        # Two separate short groups, neither long enough: must not be joined into one.
        text = "I weigh one eighty and I sleep about seven hours"
        self.assertEqual(types(text), [])

    def test_written_digits_still_match(self):
        # Mixed forms happen when one model normalizes and another does not.
        self.assertIn("PHONE", types("call me at four 0 eight 8 nine nine one six"))

    def test_multi_words_alone_never_match(self):
        self.assertEqual(types("twenty thirty forty fifty sixty seventy eighty"), [])


if __name__ == "__main__":
    unittest.main()
