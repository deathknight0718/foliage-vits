import os
import re

from pypinyin import lazy_pinyin, Style
from ..symbols import punctuation
from .tone_sandhi import ToneSandhi
from .normalizer import TextNormalizer

import jieba_fast.posseg as posseg

PATH = "/home/foliage/project/foliage-vits/data/language-opencpop-strict.txt"

pinyin_to_symbol_map = {line.split("\t")[0]: line.strip().split("\t")[1] for line in open(os.path.join(PATH)).readlines()}

rep_map = {"：": ",", "；": ",", "，": ",", "。": ".", "！": "!", "？": "?", "\n": ".", "·": ",", "、": ",", "...": "…", "$": ".", "/": ",", "—": "-", "~": "…", "～": "…"}
tone_sandhi = ToneSandhi()


def analyze(text):
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]
    phones = []
    words = []
    for sentence in sentences:
        sentence = re.sub("[a-zA-Z]+", "", sentence)
        seg_cut = posseg.lcut(sentence)
        initials = []
        finals = []
        seg_cut = tone_sandhi.pre_merge_for_modify(seg_cut)
        for word, pos in seg_cut:
            if pos == "eng":
                continue
            sub_initials = []
            sub_finals = []
            orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
            orig_finals = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
            for c, v in zip(orig_initials, orig_finals):
                sub_initials.append(c)
                sub_finals.append(v)
            sub_finals = tone_sandhi.modified_tone(word, pos, sub_finals)
            initials.append(sub_initials)
            finals.append(sub_finals)
        initials = sum(initials, [])
        finals = sum(finals, [])
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            if c == v:
                assert c in punctuation
                phone = [c]
                words.append(1)
            else:
                v_without_tone = v[:-1]
                tone = v[-1]
                pinyin = c + v_without_tone
                assert tone in "12345"
                if c:
                    # 多音节
                    v_rep_map = {"uei": "ui", "iou": "iu", "uen": "un"}
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {"v": "yu", "e": "e", "i": "y", "u": "w"}
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]
                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, sentence, raw_pinyin)
                new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                new_v = new_v + tone
                phone = [new_c, new_v]
                words.append(len(phone))
            phones += phone
    return phones, words


def normalize(text):
    # https://github.com/PaddlePaddle/PaddleSpeech/tree/develop/paddlespeech/t2s/frontend/zh_normalization
    normalizer = TextNormalizer()
    sentences = normalizer.normalize(text)
    result = ""
    for sentence in sentences:
        sentence = sentence.replace("嗯", "恩").replace("呣", "母")
        pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
        replaced_text = pattern.sub(lambda x: rep_map[x.group()], sentence)
        replaced_text = re.sub(r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text)
        result += replaced_text
    return result
