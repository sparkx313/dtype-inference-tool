import csv
import os
import sys
import tempfile
import unittest
from datetime import date, datetime, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dtypeinfer import (ColumnSampler, Settings, coerce_column, coerce_value,
                        infer_column, infer_csv, infer_table, reservoir_sample,
                        resolve_decimal_style)
from dtypeinfer.parse import (as_boolean, as_currency, as_date, as_datetime, as_email,
                              as_float, as_integer, as_ipv4, as_ipv6, as_percent,
                              as_time, as_url, as_uuid)


class TestNumbers(unittest.TestCase):
    def test_ordinary_spellings_parse(self):
        self.assertEqual(as_integer("42"), (42, "plain"))
        self.assertEqual(as_integer("+42"), (42, "plain"))
        self.assertEqual(as_integer("-7")[0], -7)
        self.assertEqual(as_float("3.5")[0], 3.5)
        self.assertEqual(as_float(".5")[0], 0.5)
        self.assertEqual(as_float("1e3")[0], 1000.0)
        self.assertEqual(as_float("1.5e-3")[0], 0.0015)

    def test_thousands_separators_are_grouped(self):
        self.assertEqual(as_integer("1,234,567"), (1234567, "thousands_separator"))
        self.assertEqual(as_float("1,234.56")[0], 1234.56)
        self.assertEqual(as_float("1,234.56")[1], "grouped_thousands")

    def test_decimal_comma_is_recognised(self):
        self.assertEqual(as_float("1234,56")[0], 1234.56)
        self.assertEqual(as_float("1234,56")[1], "decimal_comma")
        self.assertEqual(as_float("1.234,56")[0], 1234.56)

    def test_a_three_digit_tail_is_decided_by_the_column_style(self):
        # `1.234` genuinely has two readings; the dot follows the local convention
        # and the comma follows the other one, so both defaults are named.
        self.assertEqual(as_float("1.234"), (1.234, "plain"))
        self.assertEqual(as_float("1,234"), (1234.0, "integer_valued+grouped_thousands"))
        self.assertEqual(as_float("1.234", "eu"),
                         (1234.0, "integer_valued+grouped_thousands"))
        self.assertEqual(as_float("1,234", "eu"), (1.234, "decimal_comma"))

    def test_leading_zeros_are_kept_as_a_tag(self):
        self.assertEqual(as_integer("007"), (7, "leading_zero"))
        self.assertEqual(as_integer("0"), (0, "plain"))

    def test_space_grouped_numbers_parse(self):
        self.assertEqual(as_float("1 234,56")[0], 1234.56)

    def test_malformed_numbers_are_rejected(self):
        for text in ["1..2", "1,2,3,4.5.6", "--1", "1 2", "1234,567,8", "abc"]:
            with self.subTest(text=text):
                self.assertIsNone(as_float(text))

    def test_percent_and_currency(self):
        self.assertEqual(as_percent("12.5%"), (12.5, "percent"))
        self.assertEqual(as_percent("-3%")[0], -3.0)
        self.assertEqual(as_percent("(4%)")[0], -4.0)
        self.assertEqual(as_currency("$1,234.56")[0], 1234.56)
        self.assertEqual(as_currency("1.234,56 EUR")[0], 1234.56)
        self.assertEqual(as_currency("-42USD")[0], -42.0)
        self.assertIsNone(as_currency("1234"))

    def test_boolean_words_only(self):
        self.assertEqual(as_boolean("Yes"), (True, "true_word"))
        self.assertEqual(as_boolean("off")[0], False)
        # 1/0 stay integers: calling them booleans is a column-level decision.
        self.assertIsNone(as_boolean("1"))


class TestDatesAndTimes(unittest.TestCase):
    def test_iso_dates(self):
        self.assertEqual(as_date("2026-01-31"), (date(2026, 1, 31), "iso"))
        self.assertEqual(as_date("2026-1-2")[0], date(2026, 1, 2))
        self.assertEqual(as_date("20260131")[0], date(2026, 1, 31))

    def test_field_order_is_only_claimed_when_it_can_be(self):
        self.assertEqual(as_date("31/12/2020")[1], "day_first")
        self.assertEqual(as_date("12/25/2020")[1], "month_first")
        self.assertEqual(as_date("03/04/2020")[1], "order_ambiguous")
        self.assertEqual(as_date("04/04/2020")[1], "order_symmetric")

    def test_two_digit_years_are_tagged(self):
        parsed = as_date("1/2/20")
        self.assertEqual(parsed[0], date(2020, 2, 1))
        self.assertIn("two_digit_year", parsed[1])

    def test_month_names(self):
        self.assertEqual(as_date("5 Jan 2020"), (date(2020, 1, 5), "named_month"))
        self.assertEqual(as_date("January 5, 2020")[0], date(2020, 1, 5))

    def test_impossible_days_are_rejected(self):
        for text in ["2020-02-31", "2021-02-29", "13/13/2020", "2020-13-01"]:
            with self.subTest(text=text):
                self.assertIsNone(as_date(text))

    def test_times(self):
        self.assertEqual(as_time("13:05"), (time(13, 5), "hm"))
        self.assertEqual(as_time("1:05:00")[0], time(1, 5))
        self.assertEqual(as_time("12:00 AM")[0], time(0, 0))
        self.assertEqual(as_time("01:00 PM")[0], time(13, 0))
        self.assertEqual(as_time("10:00:00.123456789")[0], time(10, 0, 0, 123456))
        self.assertIn("timezone_offset_dropped", as_time("09:00:00Z")[1])

    def test_bad_times(self):
        for text in ["13:00 PM", "24:00", "25:00:00", "9:70"]:
            with self.subTest(text=text):
                self.assertIsNone(as_time(text))

    def test_datetime_needs_both_halves(self):
        self.assertEqual(as_datetime("2026-01-02T13:05:00")[0],
                         datetime(2026, 1, 2, 13, 5))
        self.assertEqual(as_datetime("2026-01-02 13:05")[0],
                         datetime(2026, 1, 2, 13, 5))
        self.assertEqual(as_datetime("5 Jan 2020 13:05")[0],
                         datetime(2020, 1, 5, 13, 5))
        self.assertIsNone(as_datetime("2026-01-02"))


class TestIdentifiers(unittest.TestCase):
    def test_uuid_only_canonical(self):
        self.assertIsNotNone(as_uuid("123e4567-e89b-12d3-a456-426614174000"))
        self.assertIsNone(as_uuid("123e4567e89b12d3a456426614174000"))
        self.assertIsNone(as_uuid("zzzz4567-e89b-12d3-a456-426614174000"))

    def test_email(self):
        for text in ["a@b.co", "first.last@sub.example.org", "tag+x@host.io"]:
            with self.subTest(text=text):
                self.assertIsNotNone(as_email(text))
        for text in ["a@b", "a@@b.co", ".a@b.co", "a@b.c", "no-at-sign", "a@b..co"]:
            with self.subTest(text=text):
                self.assertIsNone(as_email(text))

    def test_url_requires_a_scheme(self):
        self.assertEqual(as_url("http://x.com")[1], "http")
        self.assertEqual(as_url("https://a.b.example.com:8443/d?e=1#f")[0],
                         "https://a.b.example.com:8443/d?e=1#f")
        self.assertEqual(as_url("http://1.2.3.4/p")[1], "http")
        self.assertEqual(as_url("ftp://h.org:21")[1], "ftp")
        self.assertIsNone(as_url("www.x.com"))
        self.assertIsNone(as_url("://bad"))
        self.assertIsNone(as_url("http://x.com:99999/"))

    def test_bracketed_ipv6_authority(self):
        self.assertIsNotNone(as_url("http://[::1]:8080/"))
        self.assertIsNone(as_url("http://[not-an-address]:80/"))

    def test_ipv4_rejects_ambiguous_octets(self):
        self.assertIsNotNone(as_ipv4("10.0.0.1"))
        self.assertIsNone(as_ipv4("01.2.3.4"))
        self.assertIsNone(as_ipv4("10.0.0.256"))

    def test_ipv6(self):
        self.assertIsNotNone(as_ipv6("::1"))
        self.assertIsNotNone(as_ipv6("2001:db8::1"))
        self.assertIsNone(as_ipv6("1.2.3.4"))
        self.assertIsNone(as_ipv6("nope"))


class TestSampling(unittest.TestCase):
    def test_reservoir_is_bounded_and_reproducible(self):
        data = list(range(10_000))
        first = reservoir_sample(data, 100, seed=7)
        second = reservoir_sample(data, 100, seed=7)
        self.assertEqual(len(first), 100)
        self.assertEqual(first, second)
        self.assertNotEqual(first, reservoir_sample(data, 100, seed=8))

    def test_reservoir_keeps_a_small_population_whole(self):
        self.assertEqual(sorted(reservoir_sample([1, 2, 3], 100, seed=1)), [1, 2, 3])

    def test_reservoir_reaches_the_tail_of_the_input(self):
        sample = reservoir_sample(range(50_000), 200, seed=3)
        self.assertGreater(min(sample), 0)
        self.assertLess(max(sample), 50_000)
        self.assertGreater(len([value for value in sample if value > 25_000]), 0)

    def test_column_sampler_bounds_every_column_at_once(self):
        sampler = ColumnSampler(k=50, seed=0)
        for index in range(5000):
            sampler.add({"a": str(index), "b": "x"})
        self.assertEqual(sampler.rows, 5000)
        self.assertEqual(sampler.counts["a"], 5000)
        self.assertEqual(len(sampler.samples["a"]), 50)
        self.assertEqual(len(sampler.samples["b"]), 50)

    def test_zero_sample_size_is_rejected(self):
        with self.assertRaises(ValueError):
            reservoir_sample([1], 0)


class TestInferColumn(unittest.TestCase):
    def assert_type(self, values, expected, name="c"):
        result = infer_column(values, name)
        self.assertEqual(result.type, expected,
                         f"{values[:3]}... -> {result.type} fits={result.fits}")
        return result

    def test_one_case_per_type(self):
        cases = {
            "integer": ["1", "2", "3"],
            "float": ["1.5", "2.25", "3.0"],
            "percent": ["12.5%", "-3%", "0%"],
            "currency": ["$1,234.56", "$7.00", "$0.99"],
            "boolean": ["true", "FALSE", "yes", "n"],
            "date": ["2026-01-31", "2026-02-01", "2026-02-28"],
            "datetime": ["2026-01-02T13:05:00", "2026-01-02 13:05"],
            "time": ["13:05:00", "1:05 PM"],
            "uuid": ["123e4567-e89b-12d3-a456-426614174000"],
            "email": ["a@b.co", "x.y@example.org"],
            "url": ["https://qoder.com/", "http://1.2.3.4/p"],
            "ipv4": ["10.0.0.1", "192.168.1.1"],
            "ipv6": ["::1", "2001:db8::1"],
        }
        for expected, values in cases.items():
            with self.subTest(expected=expected):
                self.assert_type(values * 4, expected)

    def test_specificity_beats_generality(self):
        # All-digit compact dates also fit integer; the date reading is the more
        # specific claim, so it wins and the alternative is reported.
        result = self.assert_type(["20260131", "20260201"] * 5, "date")
        self.assertIn("also_fits:integer=1.00,float=1.00", result.notes)

    def test_a_dominant_type_with_a_few_bad_cells_keeps_its_name(self):
        values = [str(index) for index in range(30)] + ["oops"]
        result = infer_column(values, "mixed_in")
        self.assertEqual(result.type, "integer")
        self.assertAlmostEqual(result.confidence, 30 / 31, places=6)
        self.assertIn("unfit_values:1", result.notes)
        self.assertEqual(result.unfit_examples, ["oops"])

    def test_two_real_types_make_mixed(self):
        values = ["2026-01-02", "5", "2026-03-04", "7"] * 5
        result = infer_column(values, "mixed")
        self.assertEqual(result.type, "mixed")
        self.assertAlmostEqual(result.confidence, 0.5)

    def test_low_cardinality_is_categorical(self):
        values = ["MA", "TX", "CA"] * 20
        result = infer_column(values, "states")
        self.assertEqual(result.type, "categorical")
        self.assertEqual(result.distinct, 3)

    def test_prose_is_text(self):
        values = [f"the quick brown fox number {index}" for index in range(40)]
        self.assertEqual(infer_column(values, "prose").type, "text")

    def test_all_null_is_empty(self):
        result = infer_column(["", " ", None, "NA"], "nothing")
        self.assertEqual(result.type, "empty")
        self.assertEqual(result.non_null, 0)

    def test_nulls_do_not_deny_a_type(self):
        values = ["", "NA", "2026-01-01", None, "-", "2026-01-02"]
        result = infer_column(values, "sparse")
        self.assertEqual(result.type, "date")
        self.assertEqual(result.confidence, 1.0)
        self.assertIn("has_nulls:0.67", result.notes)

    def test_extra_null_tokens(self):
        values = ["\\N", "2026-01-01", "2026-01-02"]
        unnamed = infer_column(values, "c")
        self.assertEqual(unnamed.type, "date")
        self.assertAlmostEqual(unnamed.confidence, 2 / 3)
        self.assertFalse(unnamed.clean)
        result = infer_column(values, "c", null_values=("\\N",))
        self.assertEqual(result.type, "date")
        self.assertTrue(result.clean)

    def test_min_fit_decides_whether_a_claim_is_clean(self):
        values = [str(index) for index in range(20)] + ["bad", "worse"]
        strict = infer_column(values, "counts")
        lenient = infer_column(values, "counts", min_fit=0.9)
        self.assertEqual(strict.type, "integer")
        self.assertEqual(lenient.type, "integer")
        self.assertFalse(strict.clean)
        self.assertTrue(lenient.clean)
        self.assertTrue(strict.needs_review)
        self.assertFalse(lenient.needs_review)

    def test_ambiguity_notes_for_the_classic_cases(self):
        self.assertIn("boolean_like", infer_column(["0", "1"] * 10, "flag").notes)
        zip_result = infer_column(["02134", "10001", "90210"] * 5, "zip")
        self.assertIn("leading_zeros:5", zip_result.notes)
        self.assertIn("order_ambiguous:10",
                      infer_column(["03/04/2020"] * 10, "d").notes)
        self.assertIn("decimal_comma:10",
                      infer_column(["1234,56"] * 10, "eu").notes)
        self.assertIn("ambiguous_three_digit_tail:dot=10,comma=0,style=us",
                      infer_column(["1.234"] * 10, "amb").notes)

    def test_needs_review_tracks_confidence_and_ambiguity(self):
        self.assertFalse(infer_column(["1", "2", "3"] * 5, "clean").needs_review)
        self.assertTrue(infer_column(["0", "1"] * 10, "flag").needs_review)
        self.assertTrue(infer_column(["1", "x", "2", "y"], "bad").needs_review)

    def test_fits_reports_every_candidate(self):
        result = infer_column(["1", "2", "3"] * 5, "ints")
        self.assertEqual(result.fits["integer"], 1.0)
        self.assertEqual(result.fits["float"], 1.0)
        self.assertEqual(result.fits["date"], 0.0)

    def test_tags_record_the_spellings_found(self):
        result = infer_column(["$1,234.56", "12 EUR", "€3"] * 4, "money")
        self.assertEqual(result.type, "currency")
        self.assertEqual(result.tags["$:grouped_thousands"], 4)
        self.assertEqual(len(result.tags), 3)

    def test_decimal_style_can_be_forced(self):
        values = ["1.234", "2.345"]
        self.assertEqual(infer_column(values, "c").type, "float")
        forced = infer_column(values, "c", decimal_style="eu")
        self.assertEqual(forced.type, "integer")
        self.assertEqual(coerce_column(values, forced).values, [1234, 2345])

    def test_generators_are_consumed_once_and_counted(self):
        consumed = []

        def stream():
            for index in range(2000):
                consumed.append(index)
                yield str(index)

        result = infer_column(stream(), "gen", sample_size=200)
        self.assertEqual(result.population, 2000)
        self.assertEqual(result.sampled, 200)
        self.assertEqual(len(consumed), 2000)
        self.assertIn("sampled:200/2000", result.notes)

    def test_full_scan_when_sampling_is_off(self):
        result = infer_column([str(index) for index in range(1000)], "all",
                              sample_size=None)
        self.assertEqual(result.sampled, 1000)
        self.assertEqual(result.population, 1000)


class TestResolveStyle(unittest.TestCase):
    def test_one_decimal_comma_proves_eu(self):
        style, evidence = resolve_decimal_style(["1234,56", "1.234", "12,3"])
        self.assertEqual(style, "eu")
        self.assertEqual(evidence["three_digit_tail_after_dot"], 1)

    def test_conflicting_evidence_is_mixed(self):
        style, _evidence = resolve_decimal_style(["12,5", "1.5"])
        self.assertEqual(style, "mixed")

    def test_plain_columns_are_us_by_default(self):
        self.assertEqual(resolve_decimal_style(["1", "2.5"])[0], "us")


class TestCoercion(unittest.TestCase):
    def test_native_types(self):
        cases = {
            "integer": ["1", "-2"],
            "float": ["1.5"],
            "date": ["2026-01-31"],
            "time": ["13:05"],
            "boolean": ["yes"],
        }
        for type_name, values in cases.items():
            with self.subTest(type_name=type_name):
                inferred = infer_column(values * 5, type_name)
                coerced = coerce_column(values, inferred).values
                self.assertEqual(type(coerced[0]).__name__,
                                 inferred.python_type.split(".")[-1])

    def test_percent_keeps_the_number_as_written(self):
        inferred = infer_column(["12.5%"] * 5, "pct")
        self.assertEqual(coerce_column(["12.5%"], inferred).values, [12.5])

    def test_eu_style_survives_into_coercion(self):
        values = ["1234,56", "12,30"]
        inferred = infer_column(values, "eu")
        self.assertEqual(inferred.decimal_style, "eu")
        self.assertEqual(coerce_column(values, inferred).values, [1234.56, 12.3])

    def test_on_error_policies(self):
        values = ["1", "two", "3"]
        inferred = infer_column(values + ["4"] * 40, "ints")
        self.assertEqual(inferred.type, "integer")
        kept = coerce_column(values, inferred)
        self.assertEqual(kept.values, [1, "two", 3])
        self.assertEqual(kept.failed, [(1, "two")])
        self.assertEqual(coerce_column(values, inferred, on_error="null").values,
                         [1, None, 3])
        with self.assertRaises(ValueError):
            coerce_column(values, inferred, on_error="raise")

    def test_unknown_policy_is_rejected(self):
        inferred = infer_column(["1"] * 10, "ints")
        with self.assertRaises(ValueError):
            coerce_column(["1"], inferred, on_error="shrug")

    def test_coerce_value_handles_nulls(self):
        inferred = infer_column(["1", "2"] * 5, "ints")
        self.assertIsNone(coerce_value("N/A", "integer"))
        self.assertIsNone(coerce_value("", "integer"))
        self.assertEqual(coerce_value("  7 ", "integer"), 7)
        self.assertEqual(coerce_column(["", "1"], inferred).nulls, 1)


class TestTableAndCsv(unittest.TestCase):
    def test_infer_table_from_mappings(self):
        rows = [
            {"id": "1", "email": "a@b.co", "when": "2026-01-01", "amount": "$5.00"},
            {"id": "2", "email": "c@d.io", "when": "2026-01-02", "amount": "$6.00"},
        ] * 6
        table = infer_table(rows)
        self.assertEqual(table.types,
                         {"id": "integer", "email": "email", "when": "date",
                          "amount": "currency"})
        self.assertEqual(table.population, 12)
        self.assertEqual(table.summary()["columns"], 4)

    def test_infer_table_can_pick_columns(self):
        rows = [{"a": "1", "b": "zz"}, {"a": "2", "b": "qq"}] * 6
        table = infer_table(rows, columns=["a"])
        self.assertEqual(list(table.columns), ["a"])

    def test_infer_csv_reads_a_real_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sales.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["id", "placed", "total", "ship_ip", "note"])
                for index in range(500):
                    writer.writerow([
                        str(index),
                        "2026-01-%02d" % (index % 28 + 1),
                        "$%d.%02d" % (index, index % 100),
                        "10.0.0.%d" % (index % 251),
                        "" if index % 7 else None,
                    ])
            table = infer_csv(path, sample_size=200)
        self.assertEqual(table.population, 500)
        self.assertEqual(table.types["id"], "integer")
        self.assertEqual(table.types["placed"], "date")
        self.assertEqual(table.types["total"], "currency")
        self.assertEqual(table.types["ship_ip"], "ipv4")
        self.assertEqual(table.types["note"], "empty")
        self.assertIn("sampled:200/500", table.columns["id"].notes)

    def test_infer_csv_sniffs_a_semicolon_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "eu.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                handle.write("a;b\n")
                for index in range(60):
                    handle.write("%d;%s\n" % (index, "12,5"))
            table = infer_csv(path)
        self.assertEqual(table.types, {"a": "integer", "b": "float"})
        self.assertEqual(table.columns["b"].decimal_style, "eu")

    def test_infer_csv_tolerates_ragged_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "ragged.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                handle.write("a,b\n1,2,extra\n3,4\n")
            table = infer_csv(path)
        self.assertEqual(sorted(table.columns), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
