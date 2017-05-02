from bs4 import BeautifulSoup
import string
import re


def clean_text(text):
    text = remove_punctuation(text, keep_apostrophe = False) # not removing apostrophe makes NER better
    text = remove_digits(text) # remove digits
    text = text.lower() # convert to lowercase
    return text


def get_text_from_html(markup):
    # put space before '<' and after '>' to fix run-ons after removing html tags
    markup = markup.replace('<', ' <')
    markup = markup.replace('>', '> ')
    soup = BeautifulSoup(markup, 'html.parser')
    s = soup.get_text()
    return s


def remove_special_u_chars(s):
    printable = set(string.printable)
    return filter(lambda c: c in printable, s)


def remove_punctuation(s, keep_apostrophe):

    if keep_apostrophe:
        punctuation = string.punctuation.translate(None, "'")
    else:
        punctuation = string.punctuation

    regex_pattern = re.compile('[%s]' % re.escape(punctuation))
    return regex_pattern.sub(' ', s)


def remove_digits(s):
    regex_pattern = re.compile('[0-9]')
    return regex_pattern.sub(' ', s)


def remove_emails(s):
    valid_local = 'a-zA-Z0-9!#$%&\'\*\+\-\/=?^_`{|}~'
    valid_domain = 'a-zA-Z0-9\-'
    regex_string = '[^%s]([%s][%s\.]*@[%s]*\.[%s]*)[^%s]*' % (valid_local, valid_local, valid_local, valid_domain, valid_domain, valid_domain)
    regex_pattern = re.compile(regex_string)
    return regex_pattern.sub(' ', s)


def remove_web_domains(s):
    valid_url = 'a-zA-Z0-9-._~:\/?#[\]@!$&\'()*+,;=`.'
    regex_pattern_1 = re.compile('(https?|www)[%s]*' % valid_url) # remove URL's starting with http or www
    regex_pattern_2 = re.compile('[%s]*\.com' % valid_url) # remove URLs ending in .com (but no http or www)
    s = regex_pattern_1.sub(' ', s)
    s = regex_pattern_2.sub(' ', s)
    return s


def allcaps_to_lowercase(s):
    regex_pattern = re.compile(r'\b[A-Z]+\b')
    s = regex_pattern.sub(s.lower(), s)
    return s


def replace_common_synonyms(s):
    s = s.replace('career', 'job')
    return s