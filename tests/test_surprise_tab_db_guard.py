from streamlit.testing.v1 import AppTest


def test_surprise_tab_degrades_gracefully_without_db():
    """Without configured DB credentials, the page must show a message and
    stop, not crash with a raw psycopg2.OperationalError — matching every
    other DB-touching page's `db_ok` guard (see trade-journal.py)."""
    at = AppTest.from_file("pages/surprise_tab.py", default_timeout=60)
    at.run()
    assert not at.exception
