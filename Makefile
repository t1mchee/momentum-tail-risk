.PHONY: help replay live dash test register figures clean

help:
	@echo "make replay   the deterministic chain, end to end, no network and no API key"
	@echo "make live     adds the model stages; needs ANTHROPIC_API_KEY"
	@echo "make dash     the interactive read-out on http://localhost:8000"
	@echo "make test     the guard suite"
	@echo "make register re-derive REGISTER.md from project/*.yaml"

# Everything here reads committed artifacts or free public files already on disk.
# The three artifacts, in the order the README walks them. This used to run the retired
# phase's chain and end by naming reports/gate2/EXAMPLE_RISK_OUTPUT.txt as the PM page -- a
# superseded page from the first design, with a -9.47% VaR against this one's -19.29%.
replay:
	uv run python scripts/poc_e1_tail.py
	uv run python scripts/build_book_returns.py
	uv run python scripts/poc_e5_book_tail.py
	uv run python scripts/poc_e8_tail_final.py
	uv run python scripts/poc_e6_concentration.py
	uv run python scripts/poc_e9_page.py
	uv run python scripts/poc_e10_realised.py
	uv run python scripts/poc_page_pdf.py
	@echo
	@echo "replay: green. reports/poc/page_2020-10-30.txt is the PM page."

# The model seats. Each one prints its own gate rates as it goes.
live:
	@test -n "$$ANTHROPIC_API_KEY" || (echo "set ANTHROPIC_API_KEY first" && exit 1)
	uv run python scripts/gate2_extract.py loser
	uv run python scripts/gate2_extract.py winner
	uv run python scripts/gate2_debate.py
	uv run python scripts/gate2_page.py

dash:
	uv run python -m uvicorn app.server:app --port 8000

test:
	uv run python -m pytest tests/ -q

register:
	uv run python scripts/build_register_extract.py

figures:
	uv run python scripts/crash_inventory_figure.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
