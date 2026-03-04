import json
import re
from dateutil.parser import parse


def build_tokens(ocr_result):
    tokens = []
    n = len(ocr_result["text"])

    for i in range(n):
        word = ocr_result["text"][i].strip()
        if not word:
            continue

        tokens.append({
            "text": word,
            "left": ocr_result["left"][i],
            "top": ocr_result["top"][i],
            "width": ocr_result["width"][i],
            "height": ocr_result["height"][i],
            "right": ocr_result["left"][i] + ocr_result["width"][i],
            "bottom": ocr_result["top"][i] + ocr_result["height"][i]
        })

    return tokens

def group_into_lines(tokens, y_threshold=10):
    tokens = sorted(tokens, key=lambda x: (x["top"], x["left"]))

    lines = []
    current_line = []
    current_y = None

    for token in tokens:
        if current_y is None:
            current_y = token["top"]

        if abs(token["top"] - current_y) <= y_threshold:
            current_line.append(token)
        else:
            lines.append(sorted(current_line, key=lambda x: x["left"]))
            current_line = [token]
            current_y = token["top"]

    if current_line:
        lines.append(sorted(current_line, key=lambda x: x["left"]))

    joined_lines = []
    for line in lines:
        texts = [t["text"] for t in line]
        joined = " ".join(texts)
        joined_lines.append(joined)

    return joined_lines

invoice_number_pattern = r'(?:invoice\s*(?:no\.?|#|number)?|(?:no\.?|#))\s*:?\s*(\S*\d+\S*)'

date_patterns = [r'\d{4}\s*[-.]\s*(0?[1-9]|1[0-2])\s*[-.]\s*(([12]\d)|3[01]|0?[1-9])(?!\d)', #2025-07-25 or 2025.7.5
                 r'(?<!\d)(0?[1-9]|[12]\d|3[01])\s*[-.]\s*(0?[1-9]|[12]\d|3[01])\s*[-.]\s*\d{4}(?!\d)', # 31-01-2022 or 02.11.2022 or American date 01-16-2023
                 r'(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/\d{4}', #1/16/2020 or 07/13/2015
                 r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(0?[1-9]|[12]\d|3[01])(st|nd|rd|th)?,?\s+\d{4}', #apr 13, 2012 or apr 13 2012 or aug 1st, 2020
                 r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+([1-9]|[12]\d|3[01])(st|nd|rd|th)?,?\s+\d{4}', #March 11th, 2000 or March 11 2000
]

currency_before = r'[£$€¥₹]\s*\d{1,3}(?:[,\s]\d{3})+(?:[.,]\d+)?|[£$€¥₹]\s*\d+(?:[.,]\d+)?'
currency_after  = r'\d{1,3}(?:[,\s]\d{3})+(?:[.,]\d+)?\s*[£$€¥₹]|\d+(?:[.,]\d+)?\s*[£$€¥₹]'
space_thousand  = r'\d{1,3}(?:\s\d{3})+(?:[.,]\d+)?'
plain_number    = r'\d+(?:[.,]\d+)?'
no_percent      = r'(?!\s*\d*\s*%)'

money_pattern = rf'(?:{currency_before}|{currency_after}|{space_thousand}|{plain_number}){no_percent}'

clean_money_pattern = r'\d{1,3}(?:[,\s]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?'

def clean_currency(money):
    money = re.sub(r'[£$€¥₹]\s*', '', money)
    money = re.sub(r'\s*[£$€¥₹]', '', money)
    return money

def extract_invoice_number(line, lower_line):
    if "invoice" in lower_line or "#" in line:
        match = re.search(invoice_number_pattern, line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

def find_all_dates(line):
    result = []
    for pattern in date_patterns:
        all_matches = re.finditer(pattern, line, re.IGNORECASE)
        for match in all_matches:
            res = match.group(0)
            is_valid, date = get_valid_date(res)
            if is_valid:
                result.append({'text': res, 'date': date})
    return result

def get_valid_date(date_str):
    try:
        parsed = parse(date_str)
        return True, parsed
    except ValueError:
        return False, None

def extract_invoice_date(line):
    all_dates = find_all_dates(line)
    if len(all_dates)==0:
        return None
    elif len(all_dates)==1:
        return all_dates[0]['text']
    else:
        earliest = min(all_dates, key=lambda x: x["date"])
        return earliest['text']

def extract_total(line, lower_line):
    if 'total' in lower_line and 'subtotal' not in lower_line:
        all_matches = re.finditer(money_pattern, line, re.IGNORECASE)
        result = []
        for match in all_matches:
            result.append(clean_currency(match.group(0)))
        if len(result) == 0:
            return None
        elif len(result) == 1:
            return result[0]
        return max(result, key=parse_money_amount)
    return None

def extract_tax(line, lower_line):
    if 'tax' in lower_line and 'tax id' not in lower_line:
        match = re.search(money_pattern, line)
        if match:
            return clean_currency(match.group(0))
    return None

def extract_shipping(line, lower_line):
    if 'shipping' in lower_line:
        match = re.search(money_pattern, line)
        if match:
            return clean_currency(match.group(0))
    return None

def extract_variants(lines):
    invoice_numbers = []
    invoice_dates = []
    totals = []
    taxes = []
    shippings = []

    for line in lines:
        lower_line = line.lower()
        invoice_num = extract_invoice_number(line, lower_line)
        if invoice_num: invoice_numbers.append(invoice_num)

        date = extract_invoice_date(line)
        if date: invoice_dates.append(date)

        total = extract_total(line, lower_line)
        if total: totals.append(total)

        tax = extract_tax(line, lower_line)
        if tax: taxes.append(tax)

        shipping = extract_shipping(line, lower_line)
        if shipping: shippings.append(shipping)

    return invoice_numbers, invoice_dates, totals, taxes, shippings

def parse_money_amount(money):
    match = re.search(clean_money_pattern, money)
    if match:
        res = match.group(0)
        if ',' in res and '.' in res:
            # both present: comma is thousands, dot is decimal -> just remove comma
            cleaned = res.replace(',', '')  # 1,337.97 -> 1337.97
        elif re.search(r',\d{2}', res):
            # comma at end with 2 digits -> decimal separator
            cleaned = res.replace(',', '.')  # 432,50 -> 432.50
        else:
            # comma as thousands separator -> remove it
            cleaned = res.replace(',', '')  # 1,234 -> 1234

        cleaned = cleaned.replace(' ', '')
        return float(cleaned)
    else:
        return 0

def extract_invoice_data(ocr_result):

    tokens = build_tokens(ocr_result)
    lines = group_into_lines(tokens)

    invoice_numbers, invoice_dates, totals, taxes, shippings = extract_variants(lines)

    result = {}
    if len(invoice_numbers) > 0:
        result["Invoice Number"] = invoice_numbers[0]
    if len(invoice_dates) > 0:
        if len(invoice_dates) == 1:
            result["Invoice Date"] = invoice_dates[0]
        else:
            result["Invoice Date"] = min(invoice_dates, key=lambda x: parse(x))
    if len(totals) > 0:
        if len(totals) == 1:
            result["Total"] = totals[0]
        else:
            result["Total"] = max(totals, key=parse_money_amount)
    if len(taxes) > 0:
        if len(taxes) == 1:
            result["Tax"] = taxes[0]
        else:
            result["Tax"] = max(taxes, key=parse_money_amount)
    if len(shippings) > 0:
        if len(shippings) == 1:
            result["Shipping"] = shippings[0]
        else:
            result["Shipping"] = max(shippings, key=parse_money_amount)

    return result

class KeyWordsExtractor:

    def run(self, ocr_result):
        return json.dumps(extract_invoice_data(ocr_result))