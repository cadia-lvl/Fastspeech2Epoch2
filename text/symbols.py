# """ from https://github.com/keithito/tacotron """

# """
# Defines the set of symbols used in text input to the model.

# The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

# from text import cmudict, pinyin
# # import cmudict, pinyin

# _pad = "_"
# _punctuation = "!'(),.:;? "
# _special = "-"
# _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
# _silences = ["@sp", "@spn", "@sil"]

# # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
# _arpabet = ["@" + s for s in cmudict.valid_symbols]
# _pinyin = ["@" + s for s in pinyin.valid_symbols]

# # Export all symbols:
# symbols = (
#     [_pad]
#     + list(_special)
#     + list(_punctuation)
#     + list(_letters)
#     + _arpabet
#     + _pinyin
#     + _silences
# )


_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_phoneme = ['CH', 'AO1', 'OW0', 'AY0', 'EH2', 'S', 'EY2', 'EH1', 'EH0', 'UH0', 'AE1', 'AH2', 'W', 'DH', 'TH', 'UW2', 'OY1', 'P', 'T', 'AW2', 'AO2', 'AE2', 'UW1', 'AA0', 'R', 'B', 'L', 'HH', 'N', 'AA2', 'AW0', 'UH2', 'F', 'IY0', 'IH0', 'OW1', 'EY0', 'M', 'AA1', 'AW1', 'JH', 'AY1', 'Y', 'EY1', 'IY1', 'AH1', 'OW2', 'OY2', 'OY0', 'AY2', 'AE0', 'K', 'V', 'IY2', 'AO0', 'UW0', 'AH0', 'G', 'ZH', 'IH1', 'NG', 'ER0', 'SH', 'UH1', 'Z', 'D', 'ER2', 'ER1', 'IH2']
_silences = ["sp", "spn", "sil"]
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + _phoneme
    + _silences
)
# print(symbols)
# print(len(symbols))