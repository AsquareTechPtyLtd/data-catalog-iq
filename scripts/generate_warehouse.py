"""Generate the synthetic Fernwood Outfitters warehouse as a SQLite database.

Raw source tables get synthetic rows; staging, intermediate and mart tables are
then built from them with SQL, so lineage in the warehouse is real.

All data is fictional: emails use reserved example domains, phone numbers use
the 555-01xx fictional range and IP addresses use documentation ranges.

Usage: python3 scripts/generate_warehouse.py
Output: data/warehouse/warehouse.db  (table name = <schema>__<table>)
"""

import hashlib
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 42
ANCHOR = datetime(2026, 10, 1)  # "today" for the dataset
DB_PATH = Path(__file__).resolve().parents[1] / "data" / "warehouse" / "warehouse.db"

rng = random.Random(SEED)

# --- Name and value pools ------------------------------------------------------

FIRST_NAMES = [
    "Ava", "Liam", "Noah", "Mia", "Ethan", "Zoe", "Lucas", "Nora", "Mateo", "Isla",
    "Arjun", "Leila", "Kai", "Hana", "Diego", "Freya", "Idris", "Elif", "Theo", "Amara",
    "Jonah", "Sana", "Felix", "Ines", "Ravi", "Clara", "Tariq", "Yuki", "Owen", "Nadia",
    "Samir", "Lena", "Marco", "Ayla", "Hugo", "Meera", "Elias", "Maren", "Tomas", "Keiko",
]
LAST_NAMES = [
    "Abbott", "Bauer", "Castillo", "Dalton", "Eriksen", "Farouk", "Galloway", "Hale",
    "Ibarra", "Jensen", "Kowalski", "Lindgren", "Moreau", "Nakamura", "Okoye", "Pereira",
    "Quinlan", "Rasmussen", "Silva", "Tanaka", "Underhill", "Vance", "Whitaker", "Xu",
    "Yilmaz", "Zimmer", "Ashby", "Boateng", "Carver", "Duarte",
]
EMAIL_DOMAINS = ["example.com", "example.net", "example.org"]
CITIES = [  # (city, state, zip prefix, area code)
    ("Portland", "OR", "972", "503"),
    ("Denver", "CO", "802", "303"),
    ("Boise", "ID", "837", "208"),
    ("Reno", "NV", "895", "775"),
    ("Seattle", "WA", "981", "206"),
    ("Bend", "OR", "977", "541"),
    ("Salt Lake City", "UT", "841", "801"),
    ("Missoula", "MT", "598", "406"),
]
STREETS = ["Cedar", "Juniper", "Alder", "Summit", "Ridge", "Canyon", "Birch", "Willow",
           "Granite", "Lakeview", "Pine", "Aspen"]
STREET_SUFFIXES = ["St", "Ave", "Rd", "Ln", "Way", "Dr"]

CATEGORIES = {  # category: (sku code, nouns, price range)
    "tents": ("TEN", ["Dome Tent", "Backpacking Tent", "Family Tent", "Bivy"], (120, 650)),
    "sleeping_bags": ("SLP", ["Down Bag", "Synthetic Bag", "Quilt", "Liner"], (40, 420)),
    "backpacks": ("BKP", ["Daypack", "Trekking Pack", "Hydration Pack", "Duffel"], (45, 320)),
    "footwear": ("FTW", ["Hiking Boot", "Trail Runner", "Approach Shoe", "Camp Sandal"],
                 (60, 260)),
    "jackets": ("JKT", ["Rain Shell", "Puffer", "Fleece", "Softshell"], (55, 480)),
    "climbing": ("CLM", ["Harness", "Rope", "Helmet", "Chalk Bag"], (15, 300)),
    "cooking": ("CKG", ["Stove", "Cook Set", "Mug", "Water Filter"], (12, 180)),
    "accessories": ("ACC", ["Headlamp", "Trekking Poles", "Gaiters", "Beanie"], (10, 140)),
}
PRODUCT_ADJECTIVES = ["Alpine", "Ridgeline", "Basecamp", "Cascade", "Trailhead", "Summit",
                      "Timber", "Glacier", "Canyon", "Northstar"]

STORES = ["ST-PORTLAND", "ST-DENVER", "ST-BOISE", "ST-RENO", "ST-SEATTLE", "ST-BEND"]
WAREHOUSES = ["WH-RENO", "WH-DENVER", "WH-PORTLAND"]
LOYALTY_TIERS = (["Trail", "Summit", "Peak"], [70, 22, 8])

DEPARTMENTS = {  # department: (headcount, titles, salary range)
    "Retail Stores": (100, ["Sales Associate", "Cashier", "Store Manager"], (34000, 72000)),
    "Customer Support": (30, ["Support Agent", "Support Team Lead"], (42000, 78000)),
    "Engineering": (40, ["Software Engineer", "Senior Software Engineer",
                         "Engineering Manager"], (95000, 190000)),
    "Marketing": (20, ["Marketing Specialist", "Campaign Manager"], (60000, 130000)),
    "Finance": (15, ["Financial Analyst", "Accountant", "Controller"], (65000, 160000)),
    "Supply Chain": (20, ["Inventory Planner", "Warehouse Supervisor"], (50000, 115000)),
    "People": (10, ["HR Generalist", "Recruiter"], (55000, 120000)),
    "Data": (15, ["Analytics Engineer", "Data Engineer", "Data Analyst"], (85000, 175000)),
}

TICKET_TEMPLATES = [  # (subject, body)
    ("Where is my order?", "Hi, my order {order_id} has not arrived yet. Can you check the status?"),
    ("Damaged item received", "The {product} I received is damaged. Please advise on next steps."),
    ("Change delivery address",
     "Please update the delivery address on order {order_id} to {address}."),
    ("Loyalty points missing",
     "I bought something in store last week but my loyalty points are missing."),
    ("Payment failed", "My payment keeps failing at checkout. You can reach me at {phone}."),
    ("Product question", "Does the {product} come in a larger size?"),
    ("Cancel my order", "Please cancel order {order_id}, I ordered the wrong colour."),
    ("Return request",
     "I would like to return the {product}. Call me on {phone} if you need details."),
]
RETURN_REASONS = [
    "Too small", "Too large", "Arrived damaged", "Not as described",
    "Changed my mind", "Wrong item sent",
    "Wrong size, please call me on {phone} to arrange an exchange",
    "Item faulty, please collect from {address}",
]
PAGE_PATHS = ["/", "/c/tents", "/c/footwear", "/c/jackets", "/c/backpacks", "/cart",
              "/checkout", "/account", "/loyalty", "/stores", "/help/returns"]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/128.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) Safari/17.5",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5) Mobile Safari/17.5",
    "Mozilla/5.0 (Linux; Android 14) Chrome/128.0 Mobile",
]
IP_PREFIXES = ["192.0.2.", "198.51.100.", "203.0.113."]  # documentation ranges

# --- Helpers ---------------------------------------------------------------------


def ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def ds(d: date | datetime) -> str:
    return d.strftime("%Y-%m-%d")


def between(start: datetime, end: datetime) -> datetime:
    return start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds())))


def pick_weighted(options: list, weights: list):
    return rng.choices(options, weights=weights, k=1)[0]


def person_name() -> tuple[str, str]:
    return rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)


def address() -> tuple[str, str]:
    """Return (full address, postal code)."""
    city, state, zip_prefix, _ = rng.choice(CITIES)
    postal = f"{zip_prefix}{rng.randint(1, 99):02d}"
    street = f"{rng.randint(10, 9999)} {rng.choice(STREETS)} {rng.choice(STREET_SUFFIXES)}"
    return f"{street}, {city}, {state} {postal}", postal


def phone() -> str:
    area = rng.choice(CITIES)[3]
    return f"+1-{area}-555-01{rng.randint(0, 99):02d}"


def sha256(value: str | None) -> str | None:
    return hashlib.sha256(value.encode()).hexdigest() if value is not None else None


# --- Raw source rows ---------------------------------------------------------------


def gen_customers(n: int = 2000) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        first, last = person_name()
        email = f"{first}.{last}{i}@{rng.choice(EMAIL_DOMAINS)}"
        rows.append({
            "customer_id": f"C{i:06d}",
            "email": email if rng.random() < 0.1 else email.lower(),  # some mixed case
            "first_name": first,
            "last_name": last,
            "phone": phone(),
            "marketing_opt_in": int(rng.random() < 0.7),
            "created_at": ts(between(datetime(2023, 1, 1), datetime(2026, 9, 1))),
            "_address": address(),
        })
    return rows


def gen_products(n: int = 300) -> list[dict]:
    rows = []
    cats = list(CATEGORIES)
    for i in range(1, n + 1):
        cat = cats[(i - 1) % len(cats)]
        code, nouns, (lo, hi) = CATEGORIES[cat]
        rows.append({
            "product_id": f"P{i:04d}",
            "sku": f"FW-{code}-{i:04d}",
            "product_name": f"{rng.choice(PRODUCT_ADJECTIVES)} {rng.choice(nouns)}",
            "category": cat,
            "list_price_usd": round(rng.uniform(lo, hi), 2),
            "email_template_id": f"TPL-LAUNCH-{rng.randint(100, 999)}",
        })
    return rows


def gen_orders(customers: list[dict], products: list[dict], n: int = 6000):
    orders, items, transactions = [], [], []
    start = datetime(2024, 10, 1)
    end = ANCHOR - timedelta(days=1)
    item_no = 0
    for i in range(1, n + 1):
        c = rng.choice(customers)
        created = datetime.strptime(c["created_at"], "%Y-%m-%d %H:%M:%S")
        ordered = between(max(start, created), end)
        status = pick_weighted(["delivered", "shipped", "placed", "cancelled"], [80, 8, 4, 8])
        order_id = f"O{i:06d}"
        subtotal = 0.0
        for _ in range(rng.randint(1, 4)):
            item_no += 1
            p = rng.choice(products)
            qty = pick_weighted([1, 2, 3], [75, 20, 5])
            price = round(p["list_price_usd"] * rng.uniform(0.8, 1.0), 2)
            subtotal += qty * price
            items.append({
                "order_item_id": f"OI{item_no:07d}",
                "order_id": order_id,
                "product_id": p["product_id"],
                "quantity": qty,
                "unit_price_usd": price,
            })
        subtotal = round(subtotal, 2)
        discount = round(subtotal * rng.choice([0.1, 0.15, 0.2]), 2) if rng.random() < 0.25 else 0.0
        full_address, postal = c["_address"]
        orders.append({
            "order_id": order_id,
            "customer_id": c["customer_id"],
            "order_ts": ts(ordered),
            "status": status,
            "subtotal_usd": subtotal,
            "discount_usd": discount,
            "shipping_address": full_address,
            "shipping_country": "US",
            "currency": "USD",
        })
        if status in ("cancelled", "placed"):
            pay_status = "authorized"
        elif rng.random() < 0.01:
            pay_status = "failed"
        elif status == "delivered" and rng.random() < 0.06:
            pay_status = "refunded"
        else:
            pay_status = "captured"
        transactions.append({
            "transaction_id": f"TX{i:07d}",
            "order_id": order_id,
            "card_last4": f"{rng.randint(0, 9999):04d}",
            "card_brand": pick_weighted(["visa", "mastercard", "amex"], [55, 30, 15]),
            "billing_postal_code": postal,
            "amount_usd": round(subtotal - discount, 2),
            "payment_status": pay_status,
            "processed_at": ts(ordered + timedelta(minutes=rng.randint(1, 30))),
        })
    return orders, items, transactions


def gen_contacts(customers: list[dict], n: int = 1500) -> list[dict]:
    rows = []
    for i, c in enumerate(rng.sample(customers, n), start=1):
        created = datetime.strptime(c["created_at"], "%Y-%m-%d %H:%M:%S")
        email = c["email"].lower()
        if rng.random() < 0.1:  # CRM holds a different email for some customers
            email = f"{c['first_name']}.{c['last_name']}.crm{i}@example.org".lower()
        rows.append({
            "contact_id": f"K{i:05d}",
            "customer_id": c["customer_id"],
            "email": email,
            "full_name": f"{c['first_name']} {c['last_name']}",
            "date_of_birth": ds(between(datetime(1950, 1, 1), datetime(2005, 12, 31))),
            "loyalty_id": f"L{i:07d}",
            "loyalty_tier": pick_weighted(*LOYALTY_TIERS),
            "loyalty_joined_at": ds(between(created, ANCHOR - timedelta(days=1))),
        })
    return rows


def gen_tickets(customers, orders, products, n: int = 1200) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        if rng.random() < 0.9:
            c = rng.choice(customers)
            email = c["email"] if rng.random() < 0.8 else c["email"].upper()
            name = f"{c['first_name']} {c['last_name']}"
        else:
            first, last = person_name()
            email = f"{first}.{last}.guest{i}@{rng.choice(EMAIL_DOMAINS)}".lower()
            name = f"{first} {last}"
        created = between(datetime(2025, 1, 1), ANCHOR - timedelta(days=1))
        status = pick_weighted(["open", "pending", "solved", "closed"], [10, 10, 30, 50])
        first_response = (
            None if status == "open" else ts(created + timedelta(minutes=rng.randint(5, 2000)))
        )
        subject, body_template = rng.choice(TICKET_TEMPLATES)
        body = body_template.format(
            order_id=rng.choice(orders)["order_id"],
            product=rng.choice(products)["product_name"],
            address=address()[0],
            phone=phone(),
        )
        rows.append({
            "ticket_id": f"T{i:06d}",
            "requester_email": email,
            "requester_name": name,
            "subject": subject,
            "body": body,
            "priority": pick_weighted(["low", "normal", "high", "urgent"], [20, 55, 20, 5]),
            "status": status,
            "created_at": ts(created),
            "first_response_at": first_response,
        })
    return rows


def gen_campaigns(n: int = 60) -> list[dict]:
    themes = ["Spring Trail Sale", "Summer Basecamp", "Fall Layering", "Winter Summit",
              "Loyalty Double Points", "New Arrivals", "Clearance", "Gear Up Weekend"]
    rows = []
    for i in range(1, n + 1):
        channel = "email" if i <= 25 else ("paid_social" if i <= 45 else "search")
        start = between(datetime(2024, 10, 1), datetime(2026, 8, 31))
        rows.append({
            "campaign_id": f"CMP{i:03d}",
            "campaign_name": f"{rng.choice(themes)} {start.year} #{i}",
            "channel": channel,
            "start_date": ds(start),
            "end_date": ds(start + timedelta(days=rng.randint(7, 30))),
            "budget_usd": round(rng.uniform(2000, 40000), 2),
        })
    return rows


def gen_email_events(customers, campaigns) -> list[dict]:
    rows = []
    opted_in = [c for c in customers if c["marketing_opt_in"]]
    n = 0
    for cmp in (c for c in campaigns if c["channel"] == "email"):
        send_at = datetime.strptime(cmp["start_date"], "%Y-%m-%d") + timedelta(hours=9)
        for c in rng.sample(opted_in, 200):
            events = ["sent"]
            if rng.random() < 0.95:
                events.append("delivered")
                if rng.random() < 0.35:
                    events.append("opened")
                    if rng.random() < 0.25:
                        events.append("clicked")
            else:
                events.append("bounced")
            at = send_at + timedelta(minutes=rng.randint(0, 120))
            for event_type in events:
                n += 1
                at += timedelta(minutes=rng.randint(1, 600) if event_type != "sent" else 0)
                rows.append({
                    "event_id": f"E{n:07d}",
                    "recipient_email": c["email"],
                    "campaign_id": cmp["campaign_id"],
                    "event_type": event_type,
                    "event_at": ts(at),
                })
    return rows


def gen_inventory(products) -> list[dict]:
    rows, n = [], 0
    last_monday = ANCHOR - timedelta(days=ANCHOR.weekday() + 3)  # 2026-09-28
    for week in range(6):
        snap = last_monday - timedelta(weeks=5 - week)
        for p in products:
            for wh in WAREHOUSES:
                n += 1
                rows.append({
                    "snapshot_id": f"S{n:06d}",
                    "snapshot_date": ds(snap),
                    "sku": p["sku"],
                    "warehouse_id": wh,
                    "on_hand_qty": rng.randint(0, 400),
                })
    return rows


def gen_employees() -> list[dict]:
    rows, used, i = [], set(), 0
    for dept, (count, titles, (lo, hi)) in DEPARTMENTS.items():
        for _ in range(count):
            i += 1
            first, last = person_name()
            local = f"{first}.{last}".lower()
            suffix = 1
            while local in used:
                suffix += 1
                local = f"{first}.{last}{suffix}".lower()
            used.add(local)
            joined = between(datetime(2015, 1, 1), ANCHOR - timedelta(days=30))
            terminated = (
                ds(between(joined + timedelta(days=90), ANCHOR - timedelta(days=1)))
                if rng.random() < 0.15 and joined < ANCHOR - timedelta(days=120)
                else None
            )
            rows.append({
                "employee_id": f"EMP{i:04d}",
                "full_name": f"{first} {last}",
                "work_email": f"{local}@fernwood.example",
                "personal_email": f"{local}.home@example.net",
                "home_address": address()[0],
                "department": dept,
                "job_title": rng.choice(titles),
                "annual_salary": float(round(rng.uniform(lo, hi), -2)),
                "joining_date": ds(joined),
                "termination_date": terminated,
                "national_id_last4": f"{rng.randint(0, 9999):04d}",
            })
    return rows


def gen_pos(contacts, employees, n: int = 4000) -> list[dict]:
    cashiers = [e["employee_id"] for e in employees if e["department"] == "Retail Stores"]
    rows = []
    for i in range(1, n + 1):
        loyalty = rng.choice(contacts)["loyalty_id"] if rng.random() < 0.6 else None
        rows.append({
            "pos_txn_id": f"PT{i:06d}",
            "store_id": rng.choice(STORES),
            "loyalty_id": loyalty,
            "amount_usd": round(rng.uniform(8, 450), 2),
            "transacted_at": ts(between(datetime(2024, 10, 1), ANCHOR - timedelta(seconds=1))),
            "cashier_employee_id": rng.choice(cashiers),
        })
    return rows


def gen_page_views(customers, products, sessions: int = 3000) -> list[dict]:
    rows, n = [], 0
    for s in range(1, sessions + 1):
        customer = rng.choice(customers)["customer_id"] if rng.random() < 0.4 else None
        ip = f"{rng.choice(IP_PREFIXES)}{rng.randint(1, 254)}"
        agent = rng.choice(USER_AGENTS)
        anonymous_id = f"anon-{rng.randint(10**7, 10**8 - 1)}"
        at = between(ANCHOR - timedelta(days=90), ANCHOR - timedelta(hours=1))
        for _ in range(rng.randint(1, 6)):
            n += 1
            path = (
                f"/p/{rng.choice(products)['product_id']}"
                if rng.random() < 0.35
                else rng.choice(PAGE_PATHS)
            )
            rows.append({
                "view_id": f"V{n:07d}",
                "session_id": f"SES{s:06d}",
                "anonymous_id": anonymous_id,
                "customer_id": customer,
                "page_path": path,
                "ip_address": ip,
                "user_agent": agent,
                "viewed_at": ts(at),
            })
            at += timedelta(seconds=rng.randint(10, 300))
    return rows


def gen_returns(orders, transactions, customers, n: int = 800) -> list[dict]:
    by_customer = {c["customer_id"]: c for c in customers}
    payment = {t["order_id"]: t for t in transactions}
    refunded = [o for o in orders if payment[o["order_id"]]["payment_status"] == "refunded"]
    delivered = [o for o in orders if o["status"] == "delivered"
                 and payment[o["order_id"]]["payment_status"] == "captured"]
    chosen = refunded + rng.sample(delivered, n - len(refunded))
    rows = []
    for i, o in enumerate(chosen, start=1):
        c = by_customer[o["customer_id"]]
        email = c["email"].lower() if rng.random() < 0.85 else (
            f"{c['first_name']}.returns{i}@example.com".lower()
        )
        pay = payment[o["order_id"]]
        ordered = datetime.strptime(o["order_ts"], "%Y-%m-%d %H:%M:%S")
        rows.append({
            "return_id": f"R{i:05d}",
            "order_id": o["order_id"],
            "customer_id": o["customer_id"],
            "contact_email": email,
            "reason_text": rng.choice(RETURN_REASONS).format(phone=phone(), address=address()[0]),
            "refund_amount_usd": pay["amount_usd"] if pay["payment_status"] == "refunded" else 0.0,
            "requested_at": ts(min(ordered + timedelta(days=rng.randint(3, 40)),
                                   ANCHOR - timedelta(hours=1))),
        })
    return rows


# --- Schema ------------------------------------------------------------------------

DDL = {
    # Sources
    "raw_storefront__customers": """customer_id TEXT PRIMARY KEY, email TEXT, first_name TEXT,
        last_name TEXT, phone TEXT, marketing_opt_in INTEGER, created_at TEXT""",
    "raw_storefront__orders": """order_id TEXT PRIMARY KEY, customer_id TEXT, order_ts TEXT,
        status TEXT, subtotal_usd REAL, discount_usd REAL, shipping_address TEXT,
        shipping_country TEXT, currency TEXT""",
    "raw_storefront__order_items": """order_item_id TEXT PRIMARY KEY, order_id TEXT,
        product_id TEXT, quantity INTEGER, unit_price_usd REAL""",
    "raw_storefront__products": """product_id TEXT PRIMARY KEY, sku TEXT, product_name TEXT,
        category TEXT, list_price_usd REAL, email_template_id TEXT""",
    "raw_payments__transactions": """transaction_id TEXT PRIMARY KEY, order_id TEXT,
        card_last4 TEXT, card_brand TEXT, billing_postal_code TEXT, amount_usd REAL,
        payment_status TEXT, processed_at TEXT""",
    "raw_crm__contacts": """contact_id TEXT PRIMARY KEY, customer_id TEXT, email TEXT,
        full_name TEXT, date_of_birth TEXT, loyalty_id TEXT, loyalty_tier TEXT,
        loyalty_joined_at TEXT""",
    "raw_helpdesk__tickets": """ticket_id TEXT PRIMARY KEY, requester_email TEXT,
        requester_name TEXT, subject TEXT, body TEXT, priority TEXT, status TEXT,
        created_at TEXT, first_response_at TEXT""",
    "raw_marketing__email_events": """event_id TEXT PRIMARY KEY, recipient_email TEXT,
        campaign_id TEXT, event_type TEXT, event_at TEXT""",
    "raw_marketing__campaigns": """campaign_id TEXT PRIMARY KEY, campaign_name TEXT,
        channel TEXT, start_date TEXT, end_date TEXT, budget_usd REAL""",
    "raw_warehouse__inventory_snapshots": """snapshot_id TEXT PRIMARY KEY, snapshot_date TEXT,
        sku TEXT, warehouse_id TEXT, on_hand_qty INTEGER""",
    "raw_stores__pos_transactions": """pos_txn_id TEXT PRIMARY KEY, store_id TEXT,
        loyalty_id TEXT, amount_usd REAL, transacted_at TEXT, cashier_employee_id TEXT""",
    "raw_hr__employees": """employee_id TEXT PRIMARY KEY, full_name TEXT, work_email TEXT,
        personal_email TEXT, home_address TEXT, department TEXT, job_title TEXT,
        annual_salary REAL, joining_date TEXT, termination_date TEXT,
        national_id_last4 TEXT""",
    "raw_web__page_views": """view_id TEXT PRIMARY KEY, session_id TEXT, anonymous_id TEXT,
        customer_id TEXT, page_path TEXT, ip_address TEXT, user_agent TEXT, viewed_at TEXT""",
    "raw_returns__return_requests": """return_id TEXT PRIMARY KEY, order_id TEXT,
        customer_id TEXT, contact_email TEXT, reason_text TEXT, refund_amount_usd REAL,
        requested_at TEXT""",
    # Staging model materialized as a table: it hashes with sha256(), which only
    # exists in the generator's connection, so a view would fail for other clients.
    "staging__stg_marketing__email_events": """event_id TEXT PRIMARY KEY,
        recipient_email_sha256 TEXT, campaign_id TEXT, event_type TEXT, event_at TEXT""",
    # Intermediate
    "intermediate__int_customer_identity": """customer_id TEXT PRIMARY KEY, email TEXT,
        full_name TEXT, phone TEXT, date_of_birth TEXT, loyalty_tier TEXT,
        first_seen_at TEXT""",
    "intermediate__int_orders_enriched": """order_id TEXT PRIMARY KEY, customer_id TEXT,
        ordered_at TEXT, order_status TEXT, gross_amount_usd REAL, discount_usd REAL,
        net_amount_usd REAL, item_count INTEGER, payment_status TEXT""",
    "intermediate__int_customer_order_stats": """customer_id TEXT PRIMARY KEY,
        first_order_at TEXT, last_order_at TEXT, order_count INTEGER, net_revenue_usd REAL""",
    "intermediate__int_campaign_touches": """recipient_email_sha256 TEXT, campaign_id TEXT,
        channel TEXT, opened INTEGER, clicked INTEGER, touched_at TEXT,
        PRIMARY KEY (recipient_email_sha256, campaign_id)""",
    "intermediate__int_web_sessions": """session_id TEXT PRIMARY KEY, customer_id TEXT,
        ip_address TEXT, session_start TEXT, page_count INTEGER, landing_page TEXT""",
    # Marts
    "marts__dim_customers": """customer_id TEXT PRIMARY KEY, email TEXT, full_name TEXT,
        phone TEXT, loyalty_tier TEXT, first_order_at TEXT, last_order_at TEXT,
        lifetime_order_count INTEGER, is_active INTEGER""",
    "marts__dim_products": """product_id TEXT PRIMARY KEY, sku TEXT, product_name TEXT,
        category TEXT, list_price_usd REAL, email_template_id TEXT, is_active INTEGER""",
    "marts__fct_orders": """order_id TEXT PRIMARY KEY, customer_id TEXT, ordered_at TEXT,
        order_status TEXT, gross_amount_usd REAL, discount_usd REAL, net_amount_usd REAL,
        item_count INTEGER, payment_status TEXT""",
    "marts__fct_order_items": """order_item_id TEXT PRIMARY KEY, order_id TEXT,
        product_id TEXT, category TEXT, quantity INTEGER, unit_price_usd REAL,
        line_amount_usd REAL""",
    "marts__fct_payments": """transaction_id TEXT PRIMARY KEY, order_id TEXT, card_last4 TEXT,
        card_brand TEXT, billing_postal_code TEXT, amount_usd REAL, payment_status TEXT,
        processed_at TEXT""",
    "marts__customer_lifetime_value": """customer_id TEXT PRIMARY KEY,
        net_revenue_to_date_usd REAL, predicted_12m_value_usd REAL, ltv_segment TEXT""",
    "marts__mart_marketing_audience": """customer_id TEXT PRIMARY KEY, email TEXT,
        first_name TEXT, ltv_segment TEXT, audience_segment TEXT,
        is_marketing_opted_in INTEGER""",
    "marts__mart_email_engagement": """recipient_email_sha256 TEXT, campaign_id TEXT,
        sends INTEGER, opens INTEGER, clicks INTEGER, last_engaged_at TEXT,
        PRIMARY KEY (recipient_email_sha256, campaign_id)""",
    "marts__mart_campaign_performance": """campaign_id TEXT PRIMARY KEY, campaign_name TEXT,
        channel TEXT, sends INTEGER, open_rate REAL, click_rate REAL,
        attributed_orders INTEGER, attributed_revenue_usd REAL""",
    "marts__mart_loyalty_members": """customer_id TEXT PRIMARY KEY, email TEXT,
        full_name TEXT, date_of_birth TEXT, loyalty_tier TEXT, loyalty_joined_at TEXT,
        in_store_spend_usd REAL""",
    "marts__mart_support_ticket_summary": """ticket_id TEXT PRIMARY KEY, customer_id TEXT,
        requester_email TEXT, requester_name TEXT, subject TEXT, body_excerpt TEXT,
        priority TEXT, ticket_status TEXT, first_response_minutes INTEGER,
        created_at TEXT""",
    "marts__mart_finance_daily_revenue": """revenue_date TEXT, channel TEXT,
        order_count INTEGER, gross_revenue_usd REAL, refunds_usd REAL,
        net_revenue_usd REAL, PRIMARY KEY (revenue_date, channel)""",
    "marts__mart_inventory_health": """sku TEXT, warehouse_id TEXT, snapshot_date TEXT,
        on_hand_qty INTEGER, weeks_of_cover REAL, sell_through_rate REAL,
        PRIMARY KEY (sku, warehouse_id)""",
    "marts__dim_employees": """employee_id TEXT PRIMARY KEY, full_name TEXT, work_email TEXT,
        personal_email TEXT, department TEXT, job_title TEXT, annual_salary REAL,
        joining_date TEXT, termination_date TEXT""",
    "marts__mart_web_sessions": """session_id TEXT PRIMARY KEY, customer_id TEXT,
        ip_address TEXT, session_start TEXT, page_count INTEGER, landing_page TEXT""",
    "marts__legacy_customer_export": """customer_id TEXT PRIMARY KEY, email TEXT,
        first_name TEXT, last_name TEXT, phone TEXT""",
}

# --- Transformations (dependency order) ----------------------------------------------
# Staging models are views (dbt convention), except stg_marketing__email_events;
# intermediate and marts are tables.

STAGING_VIEWS = {
    "staging__stg_storefront__customers": """
        SELECT customer_id, lower(email) AS email,
               substr(email, instr(email, '@') + 1) AS email_domain,
               first_name, last_name, phone,
               marketing_opt_in AS is_marketing_opted_in, created_at
        FROM raw_storefront__customers""",
    "staging__stg_storefront__orders": """
        SELECT order_id, customer_id, order_ts AS ordered_at, status AS order_status,
               subtotal_usd, discount_usd, shipping_address, shipping_country
        FROM raw_storefront__orders""",
    "staging__stg_storefront__order_items": """
        SELECT order_item_id, order_id, product_id, quantity, unit_price_usd
        FROM raw_storefront__order_items""",
    "staging__stg_storefront__products": """
        SELECT product_id, sku, product_name, category, list_price_usd, email_template_id
        FROM raw_storefront__products""",
    "staging__stg_payments__transactions": """
        SELECT transaction_id, order_id, card_last4, card_brand, billing_postal_code,
               amount_usd, payment_status, processed_at
        FROM raw_payments__transactions""",
    "staging__stg_crm__contacts": """
        SELECT contact_id, customer_id, email, full_name, date_of_birth, loyalty_id,
               loyalty_tier, loyalty_joined_at
        FROM raw_crm__contacts""",
    "staging__stg_helpdesk__tickets": """
        SELECT ticket_id, requester_email, requester_name, subject, body, priority,
               status AS ticket_status, created_at,
               CAST(round((julianday(first_response_at) - julianday(created_at)) * 1440)
                    AS INTEGER) AS first_response_minutes
        FROM raw_helpdesk__tickets""",
    "staging__stg_marketing__campaigns": """
        SELECT campaign_id, campaign_name, channel, start_date, end_date, budget_usd
        FROM raw_marketing__campaigns""",
    "staging__stg_warehouse__inventory_snapshots": """
        SELECT snapshot_id, snapshot_date, sku, warehouse_id, on_hand_qty
        FROM raw_warehouse__inventory_snapshots""",
    "staging__stg_stores__pos_transactions": """
        SELECT pos_txn_id, store_id, loyalty_id, amount_usd, transacted_at,
               cashier_employee_id
        FROM raw_stores__pos_transactions""",
    "staging__stg_hr__employees": """
        SELECT employee_id, full_name, work_email, personal_email, home_address,
               department, job_title, annual_salary, joining_date, termination_date,
               national_id_last4
        FROM raw_hr__employees""",
    "staging__stg_web__page_views": """
        SELECT view_id, session_id, customer_id, page_path, ip_address, viewed_at
        FROM raw_web__page_views""",
    "staging__stg_returns__return_requests": """
        SELECT return_id, order_id, customer_id, contact_email, reason_text,
               refund_amount_usd, requested_at
        FROM raw_returns__return_requests""",
}

MODEL_INSERTS = {
    "staging__stg_marketing__email_events": """
        SELECT event_id, sha256(lower(recipient_email)) AS recipient_email_sha256, campaign_id,
               event_type, event_at
        FROM raw_marketing__email_events""",
    "intermediate__int_customer_identity": """
        SELECT c.customer_id, c.email, c.first_name || ' ' || c.last_name AS full_name, c.phone,
               k.date_of_birth, k.loyalty_tier, c.created_at AS first_seen_at
        FROM staging__stg_storefront__customers c
        LEFT JOIN staging__stg_crm__contacts k ON k.customer_id = c.customer_id""",
    "intermediate__int_orders_enriched": """
        SELECT o.order_id, o.customer_id, o.ordered_at, o.order_status,
               o.subtotal_usd AS gross_amount_usd, o.discount_usd,
               round(o.subtotal_usd - o.discount_usd, 2) AS net_amount_usd,
               coalesce(i.item_count, 0) AS item_count, p.payment_status
        FROM staging__stg_storefront__orders o
        LEFT JOIN (SELECT order_id, sum(quantity) AS item_count
                   FROM staging__stg_storefront__order_items GROUP BY order_id) i
               ON i.order_id = o.order_id
        LEFT JOIN (SELECT order_id, payment_status, max(processed_at)
                   FROM staging__stg_payments__transactions GROUP BY order_id) p
               ON p.order_id = o.order_id""",
    "intermediate__int_customer_order_stats": """
        SELECT customer_id, min(ordered_at) AS first_order_at, max(ordered_at) AS last_order_at,
               count(*) AS order_count, round(sum(net_amount_usd), 2) AS net_revenue_usd
        FROM intermediate__int_orders_enriched
        WHERE order_status != 'cancelled'
        GROUP BY customer_id""",
    "intermediate__int_campaign_touches": """
        SELECT e.recipient_email_sha256, e.campaign_id, c.channel,
               max(e.event_type = 'opened') AS opened, max(e.event_type = 'clicked') AS clicked,
               min(e.event_at) AS touched_at
        FROM staging__stg_marketing__email_events e
        JOIN staging__stg_marketing__campaigns c ON c.campaign_id = e.campaign_id
        GROUP BY e.recipient_email_sha256, e.campaign_id""",
    "intermediate__int_web_sessions": """
        SELECT session_id, max(customer_id) AS customer_id, ip_address,
               min(viewed_at) AS session_start, count(*) AS page_count,
               page_path AS landing_page
        FROM staging__stg_web__page_views
        GROUP BY session_id""",  # SQLite: bare columns come from the min() row
    "marts__dim_customers": f"""
        SELECT i.customer_id, i.email, i.full_name, i.phone, i.loyalty_tier,
               s.first_order_at, s.last_order_at,
               coalesce(s.order_count, 0) AS lifetime_order_count,
               coalesce(s.last_order_at >= datetime('{ts(ANCHOR)}', '-365 days'), 0) AS is_active
        FROM intermediate__int_customer_identity i
        LEFT JOIN intermediate__int_customer_order_stats s ON s.customer_id = i.customer_id""",
    "marts__dim_products": """
        SELECT product_id, sku, product_name, category, list_price_usd, email_template_id,
               CAST(substr(product_id, 2) AS INTEGER) % 17 != 0 AS is_active
        FROM staging__stg_storefront__products""",
    "marts__fct_orders": """
        SELECT order_id, customer_id, ordered_at, order_status, gross_amount_usd,
               discount_usd, net_amount_usd, item_count, payment_status
        FROM intermediate__int_orders_enriched""",
    "marts__fct_order_items": """
        SELECT i.order_item_id, i.order_id, i.product_id, p.category, i.quantity,
               i.unit_price_usd, round(i.quantity * i.unit_price_usd, 2) AS line_amount_usd
        FROM staging__stg_storefront__order_items i
        JOIN marts__dim_products p ON p.product_id = i.product_id""",
    "marts__fct_payments": """
        SELECT transaction_id, order_id, card_last4, card_brand, billing_postal_code,
               amount_usd, payment_status, processed_at
        FROM staging__stg_payments__transactions""",
    "marts__customer_lifetime_value": """
        WITH rev AS (
            SELECT d.customer_id, d.is_active,
                   coalesce(sum(CASE WHEN f.order_status != 'cancelled'
                                     THEN f.net_amount_usd END), 0) AS net
            FROM marts__dim_customers d
            LEFT JOIN marts__fct_orders f ON f.customer_id = d.customer_id
            GROUP BY d.customer_id)
        SELECT customer_id, round(net, 2) AS net_revenue_to_date_usd,
               round(net * CASE WHEN is_active = 1 THEN 0.45 ELSE 0.12 END, 2)
                   AS predicted_12m_value_usd,
               CASE WHEN net >= 1500 THEN 'vip' WHEN net >= 600 THEN 'high'
                    WHEN net >= 150 THEN 'mid' ELSE 'low' END AS ltv_segment
        FROM rev""",
    "marts__mart_marketing_audience": """
        SELECT d.customer_id, d.email, s.first_name, v.ltv_segment,
               CASE WHEN v.ltv_segment IN ('vip', 'high') AND d.is_active = 0
                         THEN 'high_value_lapsed'
                    WHEN d.is_active = 1 THEN 'active_engaged'
                    ELSE 'reactivation' END AS audience_segment,
               1 AS is_marketing_opted_in
        FROM marts__dim_customers d
        JOIN staging__stg_storefront__customers s ON s.customer_id = d.customer_id
        JOIN marts__customer_lifetime_value v ON v.customer_id = d.customer_id
        WHERE s.is_marketing_opted_in = 1""",
    "marts__mart_email_engagement": """
        SELECT recipient_email_sha256, campaign_id, 1 AS sends, opened AS opens,
               clicked AS clicks,
               CASE WHEN opened = 1 OR clicked = 1 THEN touched_at END AS last_engaged_at
        FROM intermediate__int_campaign_touches""",
    "marts__mart_campaign_performance": """
        WITH cust AS (SELECT customer_id, sha256(email) AS h FROM marts__dim_customers),
        attr AS (
            SELECT t.campaign_id, count(DISTINCT f.order_id) AS orders,
                   sum(f.net_amount_usd) AS revenue
            FROM intermediate__int_campaign_touches t
            JOIN cust c ON c.h = t.recipient_email_sha256
            JOIN marts__fct_orders f ON f.customer_id = c.customer_id
             AND f.ordered_at BETWEEN t.touched_at AND datetime(t.touched_at, '+7 days')
            WHERE t.clicked = 1 AND f.order_status != 'cancelled'
            GROUP BY t.campaign_id)
        SELECT c.campaign_id, c.campaign_name, c.channel,
               count(t.recipient_email_sha256) AS sends,
               round(avg(t.opened), 4) AS open_rate, round(avg(t.clicked), 4) AS click_rate,
               coalesce(a.orders, 0) AS attributed_orders,
               round(coalesce(a.revenue, 0), 2) AS attributed_revenue_usd
        FROM staging__stg_marketing__campaigns c
        LEFT JOIN intermediate__int_campaign_touches t ON t.campaign_id = c.campaign_id
        LEFT JOIN attr a ON a.campaign_id = c.campaign_id
        GROUP BY c.campaign_id""",
    "marts__mart_loyalty_members": f"""
        SELECT d.customer_id, d.email, d.full_name, k.date_of_birth, k.loyalty_tier,
               k.loyalty_joined_at,
               round(coalesce(sum(p.amount_usd), 0), 2) AS in_store_spend_usd
        FROM staging__stg_crm__contacts k
        JOIN marts__dim_customers d ON d.customer_id = k.customer_id
        LEFT JOIN staging__stg_stores__pos_transactions p
               ON p.loyalty_id = k.loyalty_id
              AND p.transacted_at >= datetime('{ts(ANCHOR)}', '-12 months')
        GROUP BY d.customer_id""",
    "marts__mart_support_ticket_summary": """
        SELECT t.ticket_id, d.customer_id, t.requester_email, d.full_name AS requester_name,
               t.subject, substr(t.body, 1, 200) AS body_excerpt, t.priority, t.ticket_status,
               t.first_response_minutes, t.created_at
        FROM staging__stg_helpdesk__tickets t
        LEFT JOIN marts__dim_customers d ON d.email = lower(t.requester_email)""",
    "marts__mart_finance_daily_revenue": """
        WITH online AS (
            SELECT date(ordered_at) AS d, count(*) AS n, sum(gross_amount_usd) AS g,
                   sum(net_amount_usd) AS net
            FROM marts__fct_orders WHERE order_status != 'cancelled' GROUP BY 1),
        refunds AS (
            SELECT date(processed_at) AS d, sum(amount_usd) AS r
            FROM marts__fct_payments WHERE payment_status = 'refunded' GROUP BY 1),
        store AS (
            SELECT date(transacted_at) AS d, count(*) AS n, sum(amount_usd) AS g
            FROM staging__stg_stores__pos_transactions GROUP BY 1)
        SELECT o.d AS revenue_date, 'online' AS channel, o.n AS order_count,
               round(o.g, 2) AS gross_revenue_usd, round(coalesce(r.r, 0), 2) AS refunds_usd,
               round(o.net - coalesce(r.r, 0), 2) AS net_revenue_usd
        FROM online o LEFT JOIN refunds r ON r.d = o.d
        UNION ALL
        SELECT d, 'store', n, round(g, 2), 0, round(g, 2) FROM store""",
    "marts__mart_inventory_health": f"""
        WITH latest AS (
            SELECT sku, warehouse_id, snapshot_date, on_hand_qty
            FROM staging__stg_warehouse__inventory_snapshots
            WHERE snapshot_date = (SELECT max(snapshot_date)
                                   FROM staging__stg_warehouse__inventory_snapshots)),
        sold AS (
            SELECT p.sku, sum(i.quantity) / {len(WAREHOUSES)}.0 AS units
            FROM marts__fct_order_items i
            JOIN marts__dim_products p ON p.product_id = i.product_id
            JOIN marts__fct_orders o ON o.order_id = i.order_id
            WHERE o.ordered_at >= datetime('{ts(ANCHOR)}', '-28 days')
              AND o.order_status != 'cancelled'
            GROUP BY p.sku)
        SELECT l.sku, l.warehouse_id, l.snapshot_date, l.on_hand_qty,
               round(l.on_hand_qty / nullif(s.units / 4.0, 0), 1) AS weeks_of_cover,
               round(coalesce(s.units, 0) / nullif(coalesce(s.units, 0) + l.on_hand_qty, 0), 4)
                   AS sell_through_rate
        FROM latest l LEFT JOIN sold s ON s.sku = l.sku""",
    "marts__dim_employees": """
        SELECT employee_id, full_name, work_email, personal_email, department, job_title,
               annual_salary, joining_date, termination_date
        FROM staging__stg_hr__employees""",
    "marts__mart_web_sessions": """
        SELECT session_id, customer_id, ip_address, session_start, page_count, landing_page
        FROM intermediate__int_web_sessions""",
    "marts__legacy_customer_export": """
        SELECT customer_id, email, first_name, last_name, phone
        FROM staging__stg_storefront__customers
        WHERE created_at < '2025-07-01'""",
}


# --- Build ---------------------------------------------------------------------------


def insert_rows(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    cols = [c for c in rows[0] if not c.startswith("_")]
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    conn.executemany(sql, [[r[c] for c in cols] for r in rows])


def main() -> None:
    customers = gen_customers()
    products = gen_products()
    orders, order_items, transactions = gen_orders(customers, products)
    contacts = gen_contacts(customers)
    campaigns = gen_campaigns()
    employees = gen_employees()
    raw = {
        "raw_storefront__customers": customers,
        "raw_storefront__orders": orders,
        "raw_storefront__order_items": order_items,
        "raw_storefront__products": products,
        "raw_payments__transactions": transactions,
        "raw_crm__contacts": contacts,
        "raw_helpdesk__tickets": gen_tickets(customers, orders, products),
        "raw_marketing__email_events": gen_email_events(customers, campaigns),
        "raw_marketing__campaigns": campaigns,
        "raw_warehouse__inventory_snapshots": gen_inventory(products),
        "raw_stores__pos_transactions": gen_pos(contacts, employees),
        "raw_hr__employees": employees,
        "raw_web__page_views": gen_page_views(customers, products),
        "raw_returns__return_requests": gen_returns(orders, transactions, customers),
    }

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    DB_PATH.unlink(missing_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.create_function("sha256", 1, sha256, deterministic=True)

    for table, columns in DDL.items():
        conn.execute(f"CREATE TABLE {table} ({columns})")
    for table, rows in raw.items():
        insert_rows(conn, table, rows)
    for view, select in STAGING_VIEWS.items():
        conn.execute(f"CREATE VIEW {view} AS {select}")
    for table, select in MODEL_INSERTS.items():
        expected = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
        produced = [d[0] for d in conn.execute(f"SELECT * FROM ({select}) LIMIT 0").description]
        if produced != expected:
            raise ValueError(f"{table}: SELECT yields {produced}, table has {expected}")
        conn.execute(f"INSERT INTO {table} ({', '.join(expected)}) {select}")
    conn.commit()

    objects = conn.execute(
        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') ORDER BY name"
    ).fetchall()
    for name, kind in objects:
        count = conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"{name:<45} {kind:<5} {count:>7,}")
    conn.close()
    print(f"\nwrote {DB_PATH.relative_to(DB_PATH.parents[2])}")


if __name__ == "__main__":
    main()
