from languages.zh import analyzer as zh_analyzer


def analyze(text, language="zh"):
    analyzer = zh_analyzer
    normalized_text = analyzer.normalize(text)
    phones, word2ph = analyzer.g2p(normalized_text)


if __name__ == "__main__":
    text = analyze("我丢雷")
