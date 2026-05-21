# -*- coding: utf-8 -*-

import re
import string
from typing import List, Optional


class TextUtils:

    def __init__(self):
        pass

    @staticmethod
    def valid_text(
            input_string: str,
            default_string: str = "",
    ) -> str:
        punc_remover = str.maketrans("", "", string.punctuation)  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        space_remover = str.maketrans("", "", string.whitespace)  # " \t\n\r\v\f"

        input_string_clear = input_string.translate(punc_remover).strip()  # Remove all punctuations
        input_string_clear = input_string_clear.translate(space_remover).strip()  # Remove all whitespaces

        if len(input_string_clear) > 0:
            return input_string
        else:
            return default_string

    @staticmethod
    def parse_string_to_list(
            input_string: str,
            starts_with: Optional[str] = None,
            ends_with: Optional[str] = None,
            do_lower: bool = False,
            add_start: Optional[str] = None,
            add_end: Optional[str] = None,
    ) -> List[str]:
        pattern = r"^\d+\."

        input_string = input_string.strip()
        if len(input_string) > 0:
            str_list = input_string.split("\n")
            str_list_clear = []
            for cur_info in str_list:
                cur_info = cur_info.strip()
                if len(cur_info) == 0:
                    continue
                if cur_info.startswith("-"):  # If each info line starts with somthing like "- "
                    cur_info = cur_info.lstrip("-").strip()
                    str_list_clear.append(cur_info)
                elif cur_info.startswith("*"):  # If each info line starts with somthing like "* "
                    cur_info = cur_info.lstrip("*").strip()
                    str_list_clear.append(cur_info)
                elif re.match(pattern, cur_info):  # If each info line starts with somthing like "1. "
                    cur_info = re.sub(pattern, "", cur_info)
                    str_list_clear.append(cur_info.strip())
                else:
                    str_list_clear.append(cur_info)

            str_list_clear = [_info.strip() for _info in str_list_clear if len(_info.strip()) > 0]

            # Filter out strings in `str_list_clear` that do not start/end with the specified substring
            if isinstance(starts_with, str) and len(starts_with) > 0:
                str_list_clear = [_info for _info in str_list_clear if _info.lower().startswith(starts_with)]
            if isinstance(ends_with, str) and len(ends_with) > 0:
                str_list_clear = [_info for _info in str_list_clear if _info.lower().endswith(ends_with)]

            # Add the specified substring to the start/end of the strings in `str_list_clear`
            if isinstance(add_end, str) and len(add_end) > 0:
                str_list_clear_new = []
                for _info in str_list_clear:
                    if _info.lower().endswith(add_end):
                        str_list_clear_new.append(_info.strip())
                    else:
                        str_list_clear_new.append((_info.strip() + add_end).strip())
                str_list_clear = str_list_clear_new
            if isinstance(add_start, str) and len(add_start) > 0:
                str_list_clear_new = []
                for _info in str_list_clear:
                    if _info.lower().startswith(add_start):
                        str_list_clear_new.append(_info.strip())
                    else:
                        str_list_clear_new.append((add_start + _info.strip()).strip())
                str_list_clear = str_list_clear_new

            # Make the strings in `str_list_clear` lowercase
            if do_lower:
                str_list_clear = [_info.lower() for _info in str_list_clear]

            return str_list_clear
        else:
            return []
