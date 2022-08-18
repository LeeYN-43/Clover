# This list of English stop words is taken from the "Glasgow Information
# Retrieval Group". The original list can be found at
# http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words
import unicodedata


def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


ENGLISH_STOP_WORDS = frozenset([
    "a", "about", "above", "across", "actually", "after", "afterwards", "again", "against",
    "all", "almost", "alone", "along", "already", "also", "although", "always",
    "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
    "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are",
    "around", "as", "at", "back", "be", "became", "because", "become",
    "becomes", "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "bill", "both",
    "bottom", "but", "by", "call", "can", "cannot", "cant", "can't", "co", "con",
    "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "don't",
    "down", "due", "during", "each", "easy", "eg", "eight", "either", "eleven", "else",
    "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
    "everything", "everywhere", "except", "few", "fifteen", "fifty",
    "find", "fire", "first", "five", "for", "former", "formerly", "forty",
    "found", "four", "from", "further", "give",
    "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
    "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his",
    "how", "however", "hundred", "i", "ie", "if", "i'm", "i'll", "i've", "in", "inc", "indeed",
    "interest", "is", "it", "it'll", "its", "it's", "itself", "just", "keep", "last", "latter",
    "latterly", "least", "less", "like", "ltd", "made", "many", "may", "me",
    "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly",
    "much", "must", "my", "myself", "name", "namely", "neither",
    "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone",
    "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "ok", "okay", "on",
    "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our",
    "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
    "please", "put", "rather", "re", "really", "same", "see", "seem", "seemed",
    "seeming", "seems", "serious", "several", "she", "should", "show", "side",
    "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone",
    "something", "sometime", "sometimes", "somewhere", "still", "such",
    "take", "ten", "than", "thank", "thanks", "that", "that's", "the", "their", "them",
    "themselves", "then", "thence", "there", "thereafter", "thereby",
    "therefore", "therein", "thereupon", "these", "they",
    "third", "this", "those", "though", "three", "through", "throughout",
    "thru", "thus", "to", "together", "too", "top", "toward", "towards",
    "twelve", "twenty", "two", "un", "until", "up", "upon", "us",
    "very", "via", "view", "viewing", "viewer", "was", "we", "we'll", "well", "welcome",
    "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither",
    "who", "whoever", "whole", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "wont", "won't", "yet", "you", "your", "yours", "you've", "you'll", "yourself",
    "yourselves", "youtube", "going", "want", "right", "you're", "we're", "know", "gonna", "need", "bit",
    "look", "yeah", "guys", "sure", "let's", "video", "oh", "let", "today","they're", "did", "looks",
    "different", "great" , "different", "say", "um", "probably", "kind", "doesn't", "does", "maybe", "hey",
    "we've", "better", "hope", "there's", "try"])

ENGLISH_STOP_WORDS_BERT_TOKENS = frozenset([
    2048, 2049, 2052, 2053, 2054, 2055, 2057, 2058, 2059, 2060, 
    2061, 2062, 2063, 2064, 6160, 2066, 2065, 2068, 2069, 2067, 
    2071, 6168, 2073, 2074, 2070, 2076, 2041, 2077, 2079, 2081, 
    2083, 2084, 2085, 2087, 2089, 2090, 2091, 2092, 2093, 2096, 
    2097, 2102, 4150, 2105, 2106, 2107, 2108, 2112, 2113, 2114, 
    2115, 2116, 14406, 2119, 2122, 2123, 2125, 2127, 2128, 2129, 
    2130, 2135, 2138, 6235, 2139, 2144, 2145, 2149, 2150, 2151, 
    2153, 2156, 2157, 10354, 2168, 2169, 2171, 2172, 2174, 2176, 
    2178, 2179, 2180, 2182, 6279, 2183, 2195, 2196, 2197, 2200, 
    2202, 2205, 2215, 2216, 2217, 2219, 2222, 4283, 2239, 2242, 
    6343, 2247, 4297, 16584, 2256, 2261, 4312, 2265, 2274, 8419, 
    2279, 2280, 2290, 2292, 2295, 2296, 8440, 8442, 2298, 2302, 
    2306, 2307, 2310, 4364, 2320, 2323, 2327, 4376, 10523, 4385, 
    2339, 2342, 14635, 2348, 2349, 2353, 2360, 2362, 2367, 2369, 
    2370, 2378, 4426, 8529, 16726, 4445, 2404, 2408, 2411, 2412, 
    8558, 2416, 6516, 2424, 2426, 2428, 2438, 2442, 4496, 2453, 
    2467, 2468, 2469, 27046, 2471, 2481, 2488, 2498, 2500, 6600, 
    2505, 25035, 2507, 2515, 2522, 2525, 2543, 2560, 2562, 2566, 
    2572, 2582, 2588, 2589, 2593, 4661, 2619, 2625, 6737, 2646, 
    2651, 2655, 2664, 8811, 2672, 2673, 2678, 4728, 2682, 23166, 
    2702, 2738, 2763, 2785, 2790, 2802, 2809, 2821, 2823, 6920, 
    4873, 29464, 2841, 2842, 4895, 2870, 2875, 2878, 6974, 4931, 
    6987, 2894, 2917, 4971, 2941, 2947, 2978, 2987, 2993, 3005, 
    5064, 11210, 3021, 3031, 3037, 3046, 1005, 5106, 3067, 3071, 
    3081, 1037, 3087, 1041, 7188, 1045, 1049, 3100, 1055, 1056, 
    1059, 5183, 7249, 3157, 9308, 3174, 3183, 3193, 3209, 5262, 
    23709, 3246, 3251, 3262, 3272, 9444, 25828, 11501, 13557, 
    3334, 5390, 3352, 5408, 5417, 3383, 9530, 3398, 3401, 3458, 
    3504, 5564, 3531, 5595, 26090, 3568, 5620, 9731, 5659, 3634, 
    3649, 18006, 5728, 3685, 13972, 3733, 3732, 7858, 3762, 16064, 
    7880, 3809, 3815, 7929, 5886, 3839, 3849, 5921, 3875, 3904, 
    5973, 4064, 3953, 6069, 4025, 1996, 1997, 1998, 1999, 2000, 
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 
    2011, 2012, 2013, 2014, 2016, 2017, 2018, 4067, 2020, 2021, 
    2022, 2023, 2024, 2025, 2019, 2027, 2026, 2028, 2030, 2029, 
    2031, 2032, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2033, 
    2042, 2043, 2044, 2045])