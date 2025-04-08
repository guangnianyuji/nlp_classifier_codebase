import math
class TextSplitter:
    def __init__(self, min_len=310, max_len=390, max_sep_num=5):
        self.min_len = min_len
        self.max_len = max_len
        self.std_len = (self.min_len + self.max_len) // 2
        self.max_sep_num = max_sep_num

    def __call__(self, text):
        # pdb.set_trace()
        if len(text) <= self.max_len:
            return [text]
        sep_num = math.ceil(len(text) / self.std_len)
        tgt_len = std_len = len(text) // sep_num
        min_len = std_len - (self.max_len - std_len)

        # min_len, std_len = self.min_len, self.std_len
        start_idx = 0
        result = []
        idx = None
        for sep_idx in range(min(sep_num, self.max_sep_num)):
            # print(start_idx, idx, min_len, std_len, self.max_len)
            assert min_len <= std_len <= self.max_len, (min_len, std_len, self.max_len)
            if start_idx+self.max_len >= len(text):
                idx_list = [(len(text), 0)]
            else:
                idx_list = []
                for i in range(start_idx+min_len, start_idx+self.max_len):
                    if not (('\u9fff' >= text[i] >= '\u4e00') or ('z' >= text[i] >= 'a') or ('Z' >= text[i] >= 'A') or ('9' >= text[i] >= '0')):
                        idx_list.append((i, abs(i-start_idx-std_len)-(i>start_idx+std_len)/2))
            # pdb.set_trace()
            idx = min(idx_list, key=lambda x: x[1])[0] if len(idx_list) else ((sep_idx+1)*tgt_len)
            result.append(text[start_idx:idx+1])
            start_idx = idx+1
            std_len = min((sep_idx+1)*tgt_len - (idx+1) + tgt_len, self.max_len-1)
            min_len = std_len - (self.max_len - max(std_len, tgt_len))
        return result

