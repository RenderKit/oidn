# remove section numbers for subheadings
# Based on Wagner Macedo's filter.py posted at
# https://groups.google.com/forum/#!msg/pandoc-discuss/RUC-tuu_qf0/h-H3RRVt1coJ
import pandocfilters as pf

def do_filter(k, v, f, m):
    if k == "Header" and v[0] > 2:
        v[1][1].append('unnumbered')
        return pf.Header(v[0], v[1], v[2])

if __name__ == "__main__":
    pf.toJSONFilter(do_filter)
