from .symbols import cleaned_text_to_sequence

from languages.zh import analyzer as zh_analyzer


def analyze(text, language="zh"):
    analyzer = zh_analyzer
    normalized_text = analyzer.normalize(text)
    phones, words = analyzer.analyze(normalized_text)
    return cleaned_text_to_sequence(phones), words, normalized_text


if __name__ == "__main__":
    phones, words, normalized_text = analyze("我丢雷老某")
    print(phones)
