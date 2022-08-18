from __future__ import absolute_import, division, print_function, unicode_literals

from transformers import BertTokenizerFast
from transformers import BertTokenizer as HFBertTokenizer

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertTokenizer_FromPretrained(object):
    """Runs albertTokenizerFast from transformers lib
        from huggingface transformers
    """  #  add by lyn
    def __init__(self, from_pretrained, vocab_file=None, do_lower_case=True, remove_space=True, keep_accents=False, Fast=True):
        if from_pretrained is not None:
            try:
                self.tokenizer = HFBertTokenizer.from_pretrained(from_pretrained) if not Fast else BertTokenizerFast.from_pretrained(from_pretrained)
            except:
                self.tokenizer = HFBertTokenizer.from_pretrained(from_pretrained, local_files_only=True) if not Fast else BertTokenizerFast.from_pretrained(from_pretrained, local_files_only=True)
        else:
            self.tokenizer = HFBertTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case, remove_space=remove_space,
                                                 keep_accents=keep_accents) if not Fast else \
                             BertTokenizerFast(vocab_file=vocab_file, do_lower_case=do_lower_case, remove_space=remove_space,
                                                 keep_accents=keep_accents)
    def tokenize(self, input_seq, text_pair=None, add_special_tokens=True, padding=None,
                 truncation=False, max_length=None, is_split_into_words=False, return_tensors=None,
                 return_token_type_ids=True, return_attention_mask=True, return_length=False, **kwargs):
        """ Call the tokenizer directly
            tokenize -> token2ids -> add speical tokens -> build input masks -> build token type ids 
        args:
            more details about args in
            https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__
        Return:
            return BatchEncoding, including {'input_ids', 'token_type_ids', 'attention_mask'}
        """
        return self.tokenizer(input_seq, add_special_tokens=add_special_tokens, padding=padding, truncation=truncation,
                              max_length=max_length, is_split_into_words=is_split_into_words, return_tensors=return_tensors,
                              return_token_type_ids=return_token_type_ids, return_attention_mask=return_attention_mask,
                              return_length=return_length, **kwargs)

    def convert_tokens_to_ids(self, token_ids):
        return self.tokenizer.convert_tokens_to_ids(token_ids)


