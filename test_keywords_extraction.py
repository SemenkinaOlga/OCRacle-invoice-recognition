import keywords_extraction as kwe
import pytest


@pytest.mark.parametrize("line, expected", [
    ("INVOICE", None),
    ("invoice # 0000007", "0000007"),
    ("invoice #: inv-6059", "inv-6059"),
    ("invoice 37535203", "37535203"),
    ("invoice no: 13310164", "13310164"),
    ("invoice #: inv-8114 date: 2025-07-05 po:po-49113","inv-8114"),
    ("st, invoice date 10-02-2023", None),
    ("john doe electronics itd. | n vo | c e", None),
    ("# 1234na1", "1234na1"),
    ("# 16892",  "16892"),
])
def test_extract_invoice_number(line, expected):
    result = kwe.extract_invoice_number(line, line.lower())
    assert result == expected, f"Line: '{line}' -> expected {expected!r}, got {result!r}"

def dates_found(line):
    return [r['text'] for r in kwe.find_all_dates(line)]

@pytest.mark.parametrize("line, expected", [
    ("Invoice date: 2025-07-25", ["2025-07-25"]),
    ("date 2025.7.5", ["2025.7.5"]),
    ("2025-1-9", ["2025-1-9"]),
    ("2025 - 07 - 25", ["2025 - 07 - 25"]),
    ("2025-13-01", []),
    ("2025-07-32", []),
    ("31-01-2022", ["31-01-2022"]),
    ("02.11.2022", ["02.11.2022"]),
    ("01-16-2023", ["01-16-2023"]),
    ("1-9-2023", ["1-9-2023"]),
    ("13-32-2022", []),
    ("date: 07/13/2015", ["07/13/2015"]),
    ("1/16/2020", ["1/16/2020"]),
    ("07/32/2015", []),
    ("07-13/2015", []),
    ("apr 13, 2012", ["apr 13, 2012"]),
    ("apr 13 2012", ["apr 13 2012"]),
    ("aug 1st, 2020", ["aug 1st, 2020"]),
    ("jan 2nd, 2021", ["jan 2nd, 2021"]),
    ("mar 3rd, 2019", ["mar 3rd, 2019"]),
    ("mar 3rd 2019", ["mar 3rd 2019"]),
    ("nov 4th, 2018", ["nov 4th, 2018"]),
    ("APR 13, 2012", ["APR 13, 2012"]),
    ("dec 15, 2020", ["dec 15, 2020"]),
    ("xyz 13, 2012", []),
    ("May 11, 2000", ["May 11, 2000", "May 11, 2000"]),
    ("March 11 2000", ["March 11 2000"]),
    ("March 11th, 2000", ["March 11th, 2000"]),
    ("January 1st, 2025", ["January 1st, 2025"]),
    ("July 4, 1776", ["July 4, 1776"]),
    ("MARCH 11TH, 2000", ["MARCH 11TH, 2000"]),
    ("June 18, 1934", ["June 18, 1934"]),
    ("December 15, 2020", ["December 15, 2020"]),
    ("Octember 5, 2020", []),
    ("from 2025-01-01 to 2025-12-31", ["2025-01-01", "2025-12-31"]),
    ("Invoice date: March 11th, 2000 due: 2000-04-11", ["March 11th, 2000", "2000-04-11"]),
    ("", []),
    ("Total amount due: $1,234.56", []),
    ("year 2025 only", []),
])
def test_find_all_dates(line, expected):
    result = dates_found(line)
    assert len(result) == len(expected), f"Line: '{line}' -> expected return list with size of {len(expected)}, got {result}"
    for x in expected:
        assert x in result, f"Line: '{line}' -> expected to contain {x}"

@pytest.mark.parametrize("line, expected", [
    ("st, invoice date 10-02-2023", "10-02-2023"),
    ("customer town, st 12345 due date 10-16-2023", "10-16-2023"),
    ("date: 2025-07-25", "2025-07-25"),
    ("date of issue: 07/13/2015", "07/13/2015"),
    ("date of issue! o 1/16/2020", "1/16/2020"),
    ("date: jul 8, 2025", "jul 8, 2025"),
    ("date: apr 13 2012", "apr 13 2012"),
    ("invoice #: inv-8114 date: 2025-07-05 po:po-49113", "2025-07-05"),
    ("issued: 2024-01-01 due: 2024-06-15", "2024-01-01"),
    ("due: 2024-06-15 issued: 2024-01-01", "2024-01-01"),
    ("", None),
    ("Total amount due: $1,234.56", None),
    ("invoice #: inv-8114", None),
    ("john doe electronics ltd.", None),
    ( "no dates here at all", None),
])
def test_extract_invoice_date(line, expected):
    result = kwe.extract_invoice_date(line)
    assert result == expected, f"Failed for: '{line}' -> expected {expected}, got {result}"

@pytest.mark.parametrize("money, expected", [
    # empty / no match
    ("",            0),
    # plain decimals with dot
    ("$262.50", 262.50),
    ("$250.00", 250.00),
    ("$9251.14", 9251.14),
    ("$36.44", 36.44),
    ("$379.15", 379.15),
    ("1337.97", 1337.97),
    ("$10158.78", 10158.78),
    ("10 158.78", 10158.78),
    ("$3 412.35", 3412.35),
    # comma as decimal separator
    ("432,50", 432.50),
    ("43,25", 43.25),
    ("475,75", 475.75),
    ("2 475,75", 2475.75),
    ("$17085,00", 17085.00),
    ("$17 085,00", 17085.00),
    # comma as thousands separator
    ("$1,337.97", 1337.97),
    ("$8,621.10", 8621.10),
    ("8,621.10", 8621.10),
    # currency symbol variations
    ("£1,234.56", 1234.56),
    ("€99.99", 99.99),
    ("475,75€", 475.75),
])
def test_parse_money_amount(money, expected):
    result = kwe.parse_money_amount(money)
    assert result == pytest.approx(expected), f"Failed for: '{money} -> expected {expected}, got {result}'"


@pytest.mark.parametrize("line, expected", [
    ("", None),
    ("total (usd) $262.50", "262.50"),
    ("subtotal $250.00", None),
    ("subtotal: $8621.10", None),
    ("total: $9251.14", "9251.14"),
    ("total $ 432,50 $ 43,25 $ 475,75", "475,75"),
    ("10% 432,50 43,25 475,75", None),
    ("+$17085,00,", None),
    ("total: $36.44", "36.44"),
    ("2 monitor 9 $379.15 $3412.35", None),
    ("total: $10158.78", "10158.78"),
    ("total: $1,337.97", "1,337.97"),
])
def test_extract_total(line, expected):
    result = kwe.extract_total(line, line.lower())
    assert result == expected, f"Failed for: '{line}' -> expected {expected}, got {result}"

@pytest.mark.parametrize("line, expected", [
    ("sales tax (5%) $12.50", "12.50"),
    ("tax (7%): $337.71", "337.71"),
    ("tax (20%): $6.07", "6.07"),
    ("tax (7%): $662.95", "662.95"),
    ("10% 432,50 43,25 475,75", None),
    ("vat [%] net worth vat gross worth", None),
    ("total: $9251.14", None),
    ("subtotal: $8621.10", None),
    ("tax id: 123456789", None),
    ("federal tax id 98-7654321", None),
    ("", None),
])
def test_extract_tax(line, expected):
    result = kwe.extract_tax(line, line.lower())
    assert result == expected, f"Failed for: '{line}' -> expected {expected}, got {result}"

@pytest.mark.parametrize("line, expected", [
    ("shipping: $87.93", "87.93"),
    ("shipping: $2,245.17", "2,245.17"),
    ("shipping & handling: $15.00", "15.00"),
    ("shipping cost $1 234.56", "1 234.56"),
    ("free shipping", None),
    ("total: $9251.14", None),
    ("tax (7%): $337.71", None),
    ("", None),
])
def test_extract_shipping(line, expected):
    result = kwe.extract_shipping(line, line.lower())
    assert result == expected, f"Failed for: '{line}' -> expected {expected}, got {result}"