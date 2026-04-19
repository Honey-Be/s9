test:
	pytest

clean:
	rm -rfd .pytest_cache/
	rm -rfd **/__pycache__/