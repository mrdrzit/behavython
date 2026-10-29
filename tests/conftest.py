def pytest_addoption(parser):
    parser.addoption(
        "--update-expected",
        action="store_true",
        help="Overwrite tests/expected with the outputs of the current code instead of comparing against it.",
    )
