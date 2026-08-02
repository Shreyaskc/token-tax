from tokentax import cli


def test_list_tokenizers(capsys):
    rc = cli.main(["list-tokenizers"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "gpt-4" in captured.out


def test_premium_toy_corpus(capsys):
    rc = cli.main(["premium", "gpt-4", "tam_Taml", "--corpus", "toy", "--n-resamples", "199"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "PremiumReport" in captured.out
    assert "tam_Taml" in captured.out


def test_premium_unknown_language(capsys):
    rc = cli.main(["premium", "gpt-4", "not_a_lang", "--corpus", "toy"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "not in corpus" in captured.err
