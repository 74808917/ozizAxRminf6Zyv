from __future__ import unicode_literals, print_function, division
import codecs, json


def loadFromJson(filename):
    with codecs.open(filename,'r',encoding = 'utf-8') as fp:
        data = json.load(fp, strict = False)
    return data

def saveToJson(filename, data, indent=4):
    with codecs.open(filename,'w',encoding = 'utf-8') as fp:
        json.dump(data, fp, indent=indent)


class ChunkSaver():
    def __init__(self, feat_filepaths, chunk_size=-1, convert_json=True):
        self.chunk_size = chunk_size
        self.convert_json = convert_json
        self.f_handlers = {feat: open(fpath, 'w', encoding="utf-8") \
                            for feat, fpath in feat_filepaths.items()}

    def __call__(self, features, index, last_save=False):
        state = last_save or \
                (self.chunk_size > 0 and (index+1) % self.chunk_size == 0)
        if state:
            for key, value in features.items():
                if len(value) > 0:
                    for i, item in enumerate(value):
                        if self.convert_json:
                            # Ensure double quote json format
                            encoded = json.dumps(item, ensure_ascii=False)
                            self.f_handlers[key].write(encoded+'\n')
                        else:
                            self.f_handlers[key].write(item+'\n')
            [self.f_handlers[key].flush() for key, _ in features.items()]
        return state

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for fp in self.f_handlers.values():
            fp.close()


def allocate_ranges(total_lines, batch_line_size, start_line_index):
    assert total_lines is not None \
            and batch_line_size is not None \
            and start_line_index is not None, \
            f"total_lines and batch_line_size must not be None."
    total_lines = total_lines - start_line_index
    n_files = total_lines // batch_line_size
    n_remain = total_lines % batch_line_size
    index_pairs = []
    for start, end in zip(range(0, n_files), range(1, n_files+1)):
        start_index = start*batch_line_size+start_line_index
        end_index = end*batch_line_size+start_line_index
        index_pairs.append([start_index, end_index])
    if n_remain > 0:
        index_pairs[-1][-1] += n_remain
    return index_pairs


import regex
def stringify_ranges(ranges):
    if isinstance(ranges, list):
        return "_".join([str(i) for i in ranges])
    elif isinstance(ranges, str):
        return regex.sub(r"(\d+\)-(\d+)", r"\1_\2", ranges)
    else:
        return ranges

def make_ranged_name(save_key, ranges, sep, loc=-1):
    if ranges is not None:
        srange = stringify_ranges(ranges)
        keys = save_key.split(sep)
        keys.insert(loc, srange)
        save_key = sep.join(keys)
    return save_key

def make_name_search_pattern(save_key, sep, pattern, loc=-1):
    keys = save_key.split(sep)
    keys.insert(loc, pattern)
    save_key = sep.join(keys)
    return save_key
