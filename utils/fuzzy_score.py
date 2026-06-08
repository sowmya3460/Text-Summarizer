from fuzzywuzzy import fuzz


def similarity_score(original, summary):

    return fuzz.token_set_ratio(
        original,
        summary
    )