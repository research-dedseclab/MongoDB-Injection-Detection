from __future__ import annotations
import requests
import time
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
import re
from requests.exceptions import RequestException

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (+educational-research)"}

class Row(BaseModel):
    payload: str
    source: str
    context: str

# Simple pattern to catch JSON-like payloads or $-operators
# PAYLOAD_PATTERN = re.compile(r"\{.*\}|\$[a-zA-Z]+")

# Flexible match: JSON-like OR $operator 
PAYLOAD_PATTERN = re.compile(
    r"""(
        \{.*\}
        |
        \$[a-zA-Z]+
        |
        \{[^{}]*\}                                 # Any JSON-like block
        |                                          # OR
        \"?\$[a-zA-Z_]+\"?\s*:\s*[^,}\]]+           # $operator : value
        |                                          # OR
        db\.\w+\.(find|aggregate|update|delete|insert)\s*\([^)]*\)  # db.collection.find(...)
        |                                          # OR
        \"?\$where\"?\s*:\s*\"[^\"]+\"               # $where operator
        |                                          # OR
        \"?\$regex\"?\s*:\s*\"[^\"]+\"               # $regex operator
    )""",
    re.IGNORECASE | re.VERBOSE
)

# ---------- Helper ----------
def extract_payloads_from_text(text):
    return [p.strip() for p in re.findall(PAYLOAD_PATTERN, text)]

def safe_get(url):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"[!] Failed to fetch {url}: {e}")
        return None

def fetch_text(url: str) -> str:
    """Fetch plain text from a URL."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.text

def scrape_pat_mongodb() -> list[Row]:
    """PayloadsAllTheThings MongoDB/NoSQL injection section."""
    url = "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/NoSQL%20Injection/README.md"
    text = fetch_text(url)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line and PAYLOAD_PATTERN.search(line):
            rows.append(Row(payload=line, source="PayloadsAllTheThings:NoSQL README", context="README line"))
    return rows

def scrape_hacktricks_mongodb() -> list[Row]:
    url_https = "https://hacktricks.boitatech.com.br/pentesting-web/nosql-injection"
    url_http = "http://hacktricks.boitatech.com.br/pentesting-web/nosql-injection"
    rows = []
    for url in (url_https, url_http):
        try:
            html = fetch_text(url)
            soup = BeautifulSoup(html, "html.parser")
            for element in soup.select("pre, code, p, li"):
                t = element.get_text("\n").strip()
                if t and PAYLOAD_PATTERN.search(t):
                    rows.append(Row(payload=t, source="HackTricks:MongoDB (mirror)", context="snippet"))
            return rows
        except Exception as e:
            print(f"[!] HackTricks (at {url}) failed: {e}")
    return rows

def scrape_owasp_database_cs() -> list[Row]:
    url = "https://raw.githubusercontent.com/OWASP/CheatSheetSeries/master/cheatsheets/Database_Security_Cheat_Sheet.md"
    text = fetch_text(url)
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line and PAYLOAD_PATTERN.search(line):
            rows.append(Row(payload=line, source="OWASP:Database Security CS (GitHub)", context="line"))
    return rows

def scrape_hackerone_mongodb() -> list[Row]:
    """HackerOne disclosed reports mentioning MongoDB/NoSQL injection."""
    url = "https://hackerone.com/hacktivity?query=MongoDB%20injection&sort_type=latest_disclosable_activity_at&filter=type%3Aall"
    html = fetch_text(url)
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for el in soup.select("a, p, code"):
        t = el.get_text(" ").strip()
        if t and PAYLOAD_PATTERN.search(t):
            rows.append(Row(payload=t, source="HackerOne:Hacktivity", context="snippet"))
    return rows

def scrape_exploitdb() -> list[Row]:
    """Exploit-DB search for MongoDB/NoSQL injection entries."""
    search_terms = ["mongodb injection", "nosql injection"]
    rows = []
    for term in search_terms:
        url = f"https://www.exploit-db.com/search?q={term.replace(' ', '+')}"
        html = fetch_text(url)
        soup = BeautifulSoup(html, "html.parser")
        for row_el in soup.select("table tr"):
            cols = row_el.find_all("td")
            if len(cols) >= 3:
                desc = cols[2].text.strip()
                if desc and PAYLOAD_PATTERN.search(desc):
                    rows.append(Row(payload=desc, source="Exploit-DB:Search", context="description"))
        time.sleep(1.0)  # be polite
    return rows

def scrape_acunetix_nosql() -> list[Row]:
    """Scrape Acunetix blog for NoSQL injection payloads."""
    url = "https://www.acunetix.com/blog/articles/nosql-injection/"
    rows = []
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for code in soup.find_all("code"):
            text = code.text.strip()
            if text and PAYLOAD_PATTERN.search(text):
                rows.append(Row(payload=text, source="Acunetix:Blog", context="code snippet"))
    except Exception as e:
        print(f"[!] scrape_acunetix_nosql failed: {e}")
    return rows

def scrape_sensepost_nosql() -> list[Row]:
    """Scrape SensePost blog for NoSQL error-based injection payloads."""
    url = "https://sensepost.com/blog/2025/nosql-error-based-injection/"
    rows = []
    try:
        html = fetch_text(url)
        soup = BeautifulSoup(html, "html.parser")

        # Function to check and store if payload-like
        def extract_payloads(text, context):
            text = text.strip()
            if not text:
                return
            # Check for JSON-like or MongoDB $ operators
            if PAYLOAD_PATTERN.search(text):
                rows.append(Row(payload=text, source="SensePost:NoSQL Error-Based Injection", context=context))

        # Code/pre blocks
        for element in soup.select("pre, code"):
            extract_payloads(element.get_text("\n"), "code/pre block")

        # Paragraphs and list items
        for element in soup.select("p, li"):
            extract_payloads(element.get_text(" "), "paragraph/list item")

        # Any remaining inline <span> or <strong> with suspicious content
        for element in soup.select("span, strong"):
            extract_payloads(element.get_text(" "), "inline text")

    except Exception as e:
        print(f"[!] scrape_sensepost_nosql failed: {e}")

    return rows

# Helper function to fix common errors like 404 or 403
def safe_request(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  
        return response
    except RequestException as e:
        print(f"[!] Error fetching {url}: {str(e)}")
        return None

def scrape_mongodb_docs():
    url = "https://www.mongodb.com/docs/manual/reference/operator/query/"
    resp = safe_get(url)
    if not resp: return []
    return extract_payloads_from_text(resp.text)

def scrape_imperva():
    url = "https://www.imperva.com/learn/application-security/nosql-injection/"
    resp = safe_get(url)
    if not resp: return []
    soup = BeautifulSoup(resp.text, "html.parser")
    code_blocks = soup.find_all("code")
    text = "\n".join(cb.get_text() for cb in code_blocks)
    return extract_payloads_from_text(text)

def scrape_invicti():
    url = "https://www.invicti.com/learn/nosql-injection/"
    resp = safe_get(url)
    if not resp: return []
    soup = BeautifulSoup(resp.text, "html.parser")
    code_blocks = soup.find_all("code")
    text = "\n".join(cb.get_text() for cb in code_blocks)
    return extract_payloads_from_text(text)

def scrape_owasp_nosql_injection() -> list[Row]:
    """OWASP NoSQL Injection article"""
    url = "https://owasp.org/www-project-cheat-sheets/cheatsheets/NoSQL_Injection_Prevention_Cheat_Sheet.html"
    response = safe_request(url)
    if response:
        return []
    else:
        return []

def scrape_medium_mongo_injection() -> list[Row]:
    """Medium article: Understanding MongoDB Injection Attacks by Securelayer7"""
    url = "https://medium.com/@securelayer7/understanding-mongodb-injection-attacks-1f015ae632b9"
    rows = scrape_generic_blog(url, "Medium:Understanding MongoDB Injection Attacks")
    return rows

def scrape_snyk_mongo_injection() -> list[Row]:
    """Snyk article: How to Avoid MongoDB Injection Attacks"""
    url = "https://snyk.io/blog/how-to-avoid-mongodb-injection-attacks/"
    response = safe_request(url)
    if response:
        return scrape_generic_blog(url, "Snyk:How to Avoid MongoDB Injection Attacks")
    else:
        return []

def scrape_troyhunt_nosql_injection() -> list[Row]:
    """Troy Hunt's blog: NoSQL Injection – A Modern Attack on MongoDB"""
    url = "https://www.troyhunt.com/nosql-injection-a-modern-attack-on-mongodb/"
    response = safe_request(url)
    if response:
        return scrape_generic_blog(url, "Troy Hunt:NoSQL Injection – A Modern Attack on MongoDB")
    else:
        return []

def scrape_security_boulevard_mongo_injection() -> list[Row]:
    """Security Boulevard article: MongoDB Security: NoSQL Injection and Best Practices"""
    url = "https://securityboulevard.com/2022/08/mongodb-security-nosql-injection-and-best-practices/"
    response = safe_request(url)
    if response:
        return scrape_generic_blog(url, "Security Boulevard:MongoDB Security – NoSQL Injection Best Practices")
    else:
        return []

def scrape_acunetix_nosql_injection() -> list[Row]:
    """Acunetix blog: MongoDB Injection Attacks"""
    url = "https://www.acunetix.com/blog/articles/nosql-injection/"
    rows = scrape_generic_blog(url, "Acunetix:MongoDB Injection Attacks")
    return rows

def scrape_portswigger_nosql_injection() -> list[Row]:
    """PortSwigger Web Security Academy: NoSQL Injection"""
    url = "https://portswigger.net/web-security/nosql-injection"
    rows = scrape_generic_blog(url, "PortSwigger:NoSQL Injection Guide")
    return rows

# Tutorial Scrapers
def scrape_digitalocean_mongo_security() -> list[Row]:
    """DigitalOcean: How to Secure MongoDB"""
    url = "https://www.digitalocean.com/community/tutorials/how-to-secure-mongodb-on-ubuntu-20-04"
    rows = scrape_generic_blog(url, "DigitalOcean:How to Secure MongoDB")
    return rows

def scrape_mongodb_docs_security() -> list[Row]:
    """Official MongoDB Security Manual"""
    url = "https://www.mongodb.com/docs/manual/security/"
    rows = scrape_generic_blog(url, "MongoDB Docs:Security")
    return rows

def scrape_devto_nosql_injection() -> list[Row]:
    """Dev.to article: Understanding NoSQL Injection and How to Prevent It"""
    url = "https://dev.to/securelayer7/understanding-nosql-injection-and-how-to-prevent-it-39gp"
    response = safe_request(url)
    if response:
        return scrape_generic_blog(url, "Dev.to:Understanding NoSQL Injection and Prevention")
    else:
        return []

# Security Tools Scrapers
def scrape_burp_suite_nosql() -> list[Row]:
    """Burp Suite MongoDB Injection Detection"""
    url = "https://portswigger.net/burp/extensibility/integrating-burp-suite-with-mongodb-injection-detection"
    rows = scrape_generic_blog(url, "Burp Suite:MongoDB Injection Detection")
    return rows

def scrape_owasp_zap_nosql() -> list[Row]:
    """OWASP ZAP NoSQL Injection"""
    url = "https://www.zaproxy.org/docs/desktop/addons/nosql-injection/"
    rows = scrape_generic_blog(url, "OWASP ZAP:NoSQL Injection")
    return rows

# Forums & Community Discussions Scrapers
def scrape_stackoverflow_mongo_injection() -> list[Row]:
    """Stack Overflow discussions on MongoDB Injection Attacks"""
    url = "https://stackoverflow.com/questions/tagged/mongodb-injection"
    rows = scrape_generic_blog(url, "Stack Overflow:MongoDB Injection Discussions")
    return rows

def scrape_reddit_mongo_security() -> list[Row]:
    """Reddit thread on MongoDB Security in r/netsec"""
    url = "https://www.reddit.com/r/netsec/search?q=mongo+injection"
    rows = scrape_generic_blog(url, "Reddit:r/netsec MongoDB Security")
    return rows

# Books Scrapers
def scrape_security_boulevard_mongo_injection() -> list[Row]:
    """Security Boulevard article: MongoDB Security: NoSQL Injection and Best Practices"""
    url = "https://securityboulevard.com/2022/08/mongodb-security-nosql-injection-and-best-practices/"
    response = safe_request(url)
    if response:
        return scrape_generic_blog(url, "Security Boulevard:MongoDB Security – NoSQL Injection Best Practices")
    else:
        return []

# MongoDB Security Best Practices Scraper
def scrape_mongodb_security_checklist() -> list[Row]:
    """MongoDB Security Checklist"""
    url = "https://www.mongodb.com/docs/manual/security/security-checklist/"
    rows = scrape_generic_blog(url, "MongoDB Security Checklist")
    return rows

def scrape_generic_with_user_agent(url, title):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
    response = safe_request(url)
    if response:
        return scrape_generic_blog(url, title)
    else:
        return []

def scrape_generic_blog(url: str, source_name: str) -> list[Row]:
    """
    Generic scraper for blogs/tutorials containing MongoDB NoSQL injection payloads.
    Looks in <pre>, <code>, <p>, <li>, <span>, <strong> for patterns matching PAYLOAD_PATTERN.
    """
    rows = []
    try:
        html = fetch_text(url)
        soup = BeautifulSoup(html, "html.parser")

        def extract_payloads(text, context):
            text = text.strip()
            if not text:
                return
            if PAYLOAD_PATTERN.search(text):
                rows.append(Row(payload=text, source=source_name, context=context))

        # High-likelihood containers
        for element in soup.select("pre, code"):
            extract_payloads(element.get_text("\n"), "code/pre block")

        for element in soup.select("p, li"):
            extract_payloads(element.get_text(" "), "paragraph/list item")

        for element in soup.select("span, strong"):
            extract_payloads(element.get_text(" "), "inline text")

    except Exception as e:
        print(f"[!] scrape_generic_blog failed for {url}: {e}")

    return rows

def main():
    collectors = [
        ("pat", scrape_pat_mongodb),
        ("hacktricks", scrape_hacktricks_mongodb),
        ("owasp", scrape_owasp_database_cs),
        ("h1", scrape_hackerone_mongodb),
        ("exploitdb", scrape_exploitdb),
        ("acunetix", scrape_acunetix_nosql),
        ("sensepost", scrape_sensepost_nosql),
        ("mongodb_docs", scrape_mongodb_docs),
        ("imperva", scrape_imperva),
        ("invicti", scrape_invicti),
        ("owasp_nosql_injection", scrape_owasp_nosql_injection),
        ("medium_mongo_injection", scrape_medium_mongo_injection),
        ("snyk_mongo_injection", scrape_snyk_mongo_injection),
        ("troyhunt_nosql_injection", scrape_troyhunt_nosql_injection),
        ("security_boulevard_mongo_injection", scrape_security_boulevard_mongo_injection),
        ("acunetix_mongo_injection", scrape_acunetix_nosql_injection),
        ("portswigger_nosql_injection", scrape_portswigger_nosql_injection),
        ("digitalocean_mongo_security", scrape_digitalocean_mongo_security),
        ("mongodb_docs_security", scrape_mongodb_docs_security),
        ("devto_nosql_injection", scrape_devto_nosql_injection),
        ("burp_suite_nosql", scrape_burp_suite_nosql),
        ("owasp_zap_nosql", scrape_owasp_zap_nosql),
        ("stackoverflow_mongo_injection", scrape_stackoverflow_mongo_injection),
        ("reddit_mongo_security", scrape_reddit_mongo_security),
        ("mongodb_security_checklist", scrape_mongodb_security_checklist),
        #  # Generic blog scrapers
        # ("sensepost", lambda: scrape_generic_blog(
        #     "https://sensepost.com/blog/2025/nosql-error-based-injection/",
        #     "SensePost:NoSQL Error-Based Injection"
        # )),
        # ("anotherblog", lambda: scrape_generic_blog(
        #     "https://example.com/nosql-injection-guide",
        #     "Example Blog:NoSQL Injection"
        # )),
    
    ]

    all_rows = []
    for name, fn in collectors:
        try:
            print(f"[+] Scraping {name}")
            rows = fn()
            print(f"    -> {len(rows)} rows")
            all_rows.extend(rows)
        except Exception as e:
            print(f"[!] {name} failed: {e}")

    # Convert to DataFrame using model_dump to handle Pydantic objects
    df = pd.DataFrame([r.model_dump() for r in all_rows]).drop_duplicates()
    out = DATA_RAW / "raw_mongodb_payloads.csv"
    df.to_csv(out, index=False)
    print(f"[✓] Saved raw to {out} ({len(df)} rows)")

if __name__ == "__main__":
    main()
