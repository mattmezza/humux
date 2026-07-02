from voice.pipeline import clean_for_speech


def test_strips_emoji():
    assert clean_for_speech("Done 👍 ✅") == "Done"


def test_strips_urls():
    assert "http" not in clean_for_speech("See https://example.com/x for more")
    assert "www" not in clean_for_speech("Visit www.example.com now")


def test_strips_code():
    assert clean_for_speech("Run `npm install` then go") == "Run then go"
    assert clean_for_speech("Code:\n```\nx = 1\n```\ndone") == "Code:\ndone"


def test_strips_markdown_symbols():
    assert clean_for_speech("**bold** and #heading") == "bold and heading"


def test_strips_bullets():
    assert clean_for_speech("- first\n- second") == "first\nsecond"


def test_dash_separator_becomes_pause():
    assert clean_for_speech("yes — really") == "yes, really"
    assert clean_for_speech("e-mail stays") == "e-mail stays"


def test_plain_text_untouched():
    assert clean_for_speech("Hello there, how are you?") == "Hello there, how are you?"
