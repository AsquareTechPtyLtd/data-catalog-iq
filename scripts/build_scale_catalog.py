"""Build a scale-test catalog: catalog.db plus thousands of synthetic distractor data categories.

Data Category Recommender scale test (N5). A real catalog can have thousands of data categories,
so search_data_categories cannot list them all. This copies data/catalog/catalog.db and
adds distractor data categories (business object × facet, e.g. "Shipment Exceptions",
"Web Search Queries", "Order Fulfilment Audit Trail"). Many share words with
the 13 real data categories, which makes them hard negatives for keyword search. The real
data categories and every table assignment are unchanged, so the same answer key applies.

All names are invented for the fictional Fernwood Outfitters.

Usage: python3 scripts/build_scale_catalog.py [--count 5000] [--seed 7]
Output: data/catalog/catalog_scale.db (gitignored)
"""

import argparse
import random
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "catalog" / "catalog.db"
TARGET = ROOT / "data" / "catalog" / "catalog_scale.db"

# business object → (governance domain, words that describe it)
OBJECTS = {
    "Supplier": ("Supply Chain", "vendors that supply products"), "Shipment": ("Supply Chain", "outbound parcels"),
    "Freight": ("Supply Chain", "inbound freight from suppliers"), "Purchase Order": ("Supply Chain", "orders placed with suppliers"),
    "Carrier": ("Supply Chain", "delivery carriers"), "Delivery Route": ("Supply Chain", "delivery routes and stops"),
    "Warehouse Shift": ("People", "warehouse staff shifts"), "Training Course": ("People", "employee training"),
    "Recruitment Candidate": ("People", "job applicants"), "Payroll Deduction": ("People", "deductions from employee pay"),
    "Performance Review": ("People", "employee performance reviews"), "Leave Request": ("People", "employee leave"),
    "Store Fixture": ("Finance", "store fixtures and fittings"), "Store Lease": ("Finance", "store property leases"),
    "Ledger Entry": ("Finance", "general ledger postings"), "Budget Plan": ("Finance", "departmental budgets"),
    "Tax Filing": ("Finance", "sales and payroll tax filings"), "Chargeback": ("Finance", "card payment disputes"),
    "Gift Card": ("Finance", "gift card balances and redemptions"), "Refund Approval": ("Finance", "refund approvals"),
    "Order Fulfilment": ("Finance", "picking and packing customer orders"), "Price Change": ("Finance", "product price changes"),
    "Promotion": ("Marketing", "discount promotions"), "Coupon": ("Marketing", "coupon codes"),
    "Marketing Asset": ("Marketing", "images and copy for campaigns"), "Social Media Post": ("Marketing", "social posts"),
    "Affiliate Partner": ("Marketing", "affiliate partners"), "Influencer Contract": ("Marketing", "influencer deals"),
    "Media Spend": ("Marketing", "paid media spend"), "Sponsorship": ("Marketing", "event sponsorships"),
    "Survey Response": ("Customer", "customer survey answers"), "Customer Complaint": ("Customer", "complaints"),
    "Warranty Claim": ("Customer", "product warranty claims"), "Wishlist": ("Customer", "saved wishlist items"),
    "Event Registration": ("Customer", "store event sign-ups"), "Community Forum Post": ("Customer", "forum posts"),
    "Call Recording": ("Support", "recorded support calls"), "Chatbot Transcript": ("Support", "chatbot conversations"),
    "Repair Ticket": ("Support", "gear repair requests"), "Knowledge Article": ("Support", "help centre articles"),
    "Web Search Query": ("Digital", "searches on the website"), "App Session": ("Digital", "mobile app sessions"),
    "Push Notification": ("Digital", "app push notifications"), "Shopping Cart": ("Digital", "online shopping carts"),
    "Recommendation Impression": ("Digital", "product recommendations shown"), "Page Speed": ("Digital", "website performance"),
    "IT Asset": ("Digital", "laptops and devices"), "Access Log": ("Digital", "system access logs"),
    "Product Review": ("Supply Chain", "customer product reviews"), "Product Safety": ("Supply Chain", "product safety tests"),
    "Recall Notice": ("Supply Chain", "product recalls"), "Packaging Spec": ("Supply Chain", "packaging specifications"),
    "Assortment Plan": ("Supply Chain", "store assortment plans"), "Demand Forecast": ("Supply Chain", "demand forecasts"),
    "Carbon Emission": ("Finance", "carbon emissions reporting"), "Energy Usage": ("Finance", "store energy use"),
    "Rental Booking": ("Customer", "gear rental bookings"), "Fraud Alert": ("Finance", "suspected fraud on orders"),
    "Store Visit": ("Customer", "store footfall counts"), "Return Label": ("Supply Chain", "return shipping labels"),
}
FACETS = [
    "Records", "History", "Metrics", "Events", "Master Data", "Audit Trail", "Exceptions", "Forecasts", "Approvals",
    "Snapshots", "Status Changes", "Costs", "Schedules", "Contracts", "Documents", "Notes", "Ratings", "Targets",
    "Incidents", "Logs", "Attributes", "Reference Codes", "Summaries", "Benchmarks", "Requests", "Assignments",
    "Settlements", "Adjustments", "Classifications", "Compliance Checks", "Quality Scores", "Volumes", "Capacity",
    "Locations", "Hierarchies", "Mappings", "Feedback", "Archives", "Alerts", "Thresholds", "Allocations",
    "Reconciliations", "Invoices", "Payments", "Usage", "Trends", "Segments", "Profiles", "Exports", "Imports",
    "Workflows", "Tasks", "Milestones", "Risks", "Issues", "Changes", "Versions", "Releases", "Tests", "Results",
    "Measurements", "Observations", "Inspections", "Certifications", "Licences", "Permits", "Claims", "Disputes",
    "Escalations", "Resolutions", "Calendars", "Rosters", "Rates", "Prices", "Quotes", "Bids", "Tenders",
    "Agreements", "Policies", "Rules", "Configurations", "Templates", "Catalogue Entries", "Identifiers", "Tags",
]
RETENTION = [None, 3, 6, 12, 13, 24, 36, 60, 72, 84, 120]


def distractors(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pairs = [(o, f) for o in OBJECTS for f in FACETS]
    rng.shuffle(pairs)
    out = []
    for i, (obj, facet) in enumerate(pairs[:count]):
        domain, about = OBJECTS[obj]
        months = rng.choice(RETENTION)
        out.append({
            "data_category_id": f"DC-{1000 + i}", "name": f"{obj} {facet}", "domain": domain,
            "status": "Approved" if rng.random() < 0.9 else "Candidate",
            "definition": f"{facet} about {about}: the {facet.lower()} kept for each {obj.lower()}.",
            "retention_months": months, "retention_trigger": f"after {obj.lower()} closed" if months else None,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.count > len(OBJECTS) * len(FACETS):
        raise SystemExit(f"at most {len(OBJECTS) * len(FACETS)} distractors")

    shutil.copyfile(SOURCE, TARGET)
    conn = sqlite3.connect(TARGET)
    base_url = conn.execute("SELECT value FROM settings WHERE key = 'base_url'").fetchone()[0]
    rows = distractors(args.count, args.seed)
    conn.executemany("INSERT INTO data_categories VALUES (?,?,?,?,?,?,?,?,?,?)", [
        (b["data_category_id"], b["name"], b["domain"], b["status"], b["definition"], b["retention_months"],
         b["retention_trigger"], "Delete" if b["retention_months"] else None, None,
         f"{base_url}/data-category/{b['data_category_id']}")
        for b in rows])
    conn.executemany("INSERT INTO data_category_search VALUES (?,?,?,?)",
                     [(b["data_category_id"], b["name"], b["definition"], "") for b in rows])
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'data_category_vectors'").fetchone():
        from catalog_iq.tools.retrieval import write_vectors  # the copy has the 13 real ones only: embed all
        write_vectors(conn)
    conn.commit()
    total = conn.execute("SELECT count(*) FROM data_categories").fetchone()[0]
    conn.close()
    print(f"wrote {TARGET.relative_to(ROOT)}: {total} data categories ({args.count} distractors + 13 real)")


if __name__ == "__main__":
    main()
