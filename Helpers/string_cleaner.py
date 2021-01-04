import re


def word_cleaner(string):
    string = string.lower()
    string = re.sub(r"i'm", "i am", string)
    string = re.sub(r"it's", "it is", string)
    string = re.sub(r"he's", "he is", string)
    string = re.sub(r"she's", "she is", string)
    string = re.sub(r"that's", "that is", string)
    string = re.sub(r"what's", "what is", string)
    string = re.sub(r"where's", "where is", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"won't", "will not", string)
    string = re.sub(r"can't", "cannot", string)
    string = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", string)
    return string
