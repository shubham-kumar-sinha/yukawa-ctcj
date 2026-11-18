"""Parse the C_T, C_J formulas of the paper into a list of (rational, zeta3, structure) terms,
and evaluate structures numerically for explicit coupling tensors.

Structures are tuples of factors:
  ('tr', word)      word: string over letters; 'T' = current generator on fermions, other letters = y^letter
  ('lam', 'abcd')   lambda tensor with indices
  ('t', 'ab')       scalar current generator
"""
import re, itertools
from fractions import Fraction as Fr
import numpy as np
import opt_einsum as oe

import os
HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep

def _frac(tok):
    # tok like \tfrac34, \tfrac{9}{32}, \tfrac1{32}, \tfrac{11}{2}
    m = re.fullmatch(r'\\tfrac(\{\d+\}|\d)(\{\d+\}|\d)', tok)
    a, b = (x.strip('{}') for x in m.groups())
    return Fr(int(a), int(b))

def _num(expr):
    """expr: e.g. '\\tfrac34\\zeta_3-\\tfrac{9}{32}' or '\\tfrac78-3\\zeta_3' -> (rat, z3)"""
    rat, z3 = Fr(0), Fr(0)
    expr = re.sub(r'(^|[+-])\\zeta_3', r'\g<1>1\\zeta_3', expr)    # a bare \zeta_3 has coefficient 1
    for sgn, body in re.findall(r'([+-]?)((?:\\tfrac(?:\{\d+\}|\d)(?:\{\d+\}|\d)|\d+)(?:\\zeta_3)?)', expr):
        s = -1 if sgn == '-' else 1
        isz = body.endswith('\\zeta_3')
        body = body.replace('\\zeta_3', '')
        v = _frac(body) if body.startswith('\\tfrac') else Fr(int(body))
        if isz: z3 += s * v
        else: rat += s * v
    return rat, z3

def _structure(st, word_mode):
    st = st.replace('\\,', '').replace(' ', '')
    factors = []
    # lambda and t factors
    for m in re.finditer(r'\\lambda_\{(\w+)\}', st):
        factors.append(('lam', m.group(1)))
    for m in re.finditer(r't_\{(\w+)\}', st):
        factors.append(('t', m.group(1)))
    rest = re.sub(r'\\lambda_\{\w+\}', '', st)
    rest = re.sub(r't_\{\w+\}', '', rest)
    if word_mode == 'Tr':
        words = re.findall(r'\\Tr\((\w+)\)', rest)
        rest2 = re.sub(r'\\Tr\(\w+\)', '', rest)
        assert rest2 == '', (st, rest2)
    else:
        words = [w for w in rest.split('\\cdot') if w]
    for w in words:
        assert re.fullmatch(r'[A-Za-z]+', w), (st, w)
        factors.append(('tr', w))
    return tuple(factors)

def parse_block(tex, word_mode):
    s = tex
    s = s.split('={}', 1)[1]
    s = s.replace('\\nonumber', '').replace('\\\\', '').replace('&', '').replace('\n', ' ')
    s = s.replace('+\\mathcal O(\\epsilon^4).', '').replace('+\\mathcal O(\\epsilon^4)', '')
    s = s.strip()
    # split into terms at top-level + or -
    terms, depth, cur = [], 0, ''
    i = 0
    while i < len(s):
        if s.startswith('\\big(', i): depth += 1; cur += '\\big('; i += 5; continue
        if s.startswith('\\big)', i): depth -= 1; cur += '\\big)'; i += 5; continue
        c = s[i]
        if c in '+-' and depth == 0 and cur.strip() != '' and not cur.rstrip().endswith('{'):
            terms.append(cur); cur = c
        else:
            cur += c
        i += 1
    terms.append(cur)
    out = []
    for t in terms:
        t = t.strip()
        sign = -1 if t.startswith('-') else 1
        t = t.lstrip('+-').strip()
        if t.startswith('\\big('):
            j = t.index('\\big)')
            rat, z3 = _num(t[5:j]); st = t[j + 5:]
        else:
            m = re.match(r'(\\tfrac(?:\{\d+\}|\d)(?:\{\d+\}|\d)|\d+)(?:\\,)?', t)
            if m:
                tok = m.group(1)
                rat = _frac(tok) if tok.startswith('\\tfrac') else Fr(int(tok)); z3 = Fr(0)
                st = t[m.end():]
            else:
                rat, z3, st = Fr(1), Fr(0), t
        out.append((sign * rat, sign * z3, _structure(st, word_mode)))
    return out

CT3 = parse_block(open(HERE + 'ct3.tex').read(), 'Tr')
CJ3 = parse_block(open(HERE + 'cj3.tex').read(), 'word')


def is_odd(st):
    """True for the odd fermion-loop terms (traces with an odd number of Yukawa matrices): the last terms of the
    paper's C_T and C_J at O(eps^3), present only in theories without a four-dimensional counterpart."""
    return any(f[0] == 'tr' and sum(ch != 'T' for ch in f[1]) % 2 for f in st)


CT3_EVEN = [x for x in CT3 if not is_odd(x[2])]      # four-dimensional Dirac algebra only
CJ3_EVEN = [x for x in CJ3 if not is_odd(x[2])]
assert len(CT3) - len(CT3_EVEN) == 1 and len(CJ3) - len(CJ3_EVEN) == 2

# O(eps^2) formulas, typed from eqs (CT), (CJ) of the paper
CT2 = [(Fr(-5, 4), 0, (('tr', 'aa'),)),
       (Fr(-1, 6), 0, (('tr', 'ab'), ('tr', 'ab'))),
       (Fr(-11, 48), 0, (('tr', 'aabb'),)),
       (Fr(-19, 6), 0, (('tr', 'abab'),)),
       (Fr(-5, 36), 0, (('lam', 'abcd'), ('lam', 'abcd')))]
CJ2 = [(Fr(-3, 2), 0, (('tr', 'TTaa'),)),
       (Fr(-1), 0, (('tr', 'TTaabb'),)),
       (Fr(-17, 2), 0, (('tr', 'TTabab'),)),
       (Fr(-5, 8), 0, (('tr', 'TTabba'),)),
       (Fr(3, 4), 0, (('tr', 'TaTabb'),)),
       (Fr(15, 2), 0, (('tr', 'TaTbab'),)),
       (Fr(3, 2), 0, (('tr', 'TabTba'),)),
       (Fr(-1), 0, (('tr', 'TTab'), ('tr', 'ab'))),
       (Fr(3, 2), 0, (('tr', 'TaTb'), ('tr', 'ab'))),
       (Fr(-3, 8), 0, (('t', 'ae'), ('t', 'bf'), ('lam', 'abcd'), ('lam', 'efcd')))]

def nlam(st):
    return sum(1 for f in st if f[0] == 'lam')

def ny(st):
    return sum(sum(1 for ch in f[1] if ch != 'T') for f in st if f[0] == 'tr')

# ---------------- numerical evaluation with explicit tensors ----------------
def eval_structure(st, Y, LAMs, T=None, t=None):
    """Y[a,i,j]; LAMs: list of lambda tensors, one per lam factor (allows multilinear expansion)."""
    ops, subs = [], []
    letters = iter('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    li = 0
    for f in st:
        if f[0] == 'lam':
            ops.append(LAMs[li]); li += 1; subs.append(f[1])
        elif f[0] == 't':
            ops.append(t); subs.append(f[1])
        else:
            w = f[1]
            fidx = [next(letters) for _ in w]
            for k, ch in enumerate(w):
                i1, i2 = fidx[k], fidx[(k + 1) % len(w)]
                if ch == 'T':
                    ops.append(T); subs.append(i1 + i2)
                else:
                    ops.append(Y); subs.append(ch + i1 + i2)
    expr = ','.join(subs) + '->'
    return oe.contract(expr, *ops, optimize='dp')

def evaluate(formula, Ypow_fn, lam_parts, T=None, t=None):
    """Return dict mapping (ypow, lam-monomial tuple) -> (rat_part, z3_part) as floats.
    Y is fixed (unit-normalized); lam_parts: dict name->tensor; the lambda multilinear expansion is
    returned as keys of sorted tuples of names."""
    res = {}
    for rat, z3, st in formula:
        k = nlam(st)
        for combo in itertools.product(sorted(lam_parts), repeat=k):
            v = eval_structure(st, Ypow_fn, [lam_parts[c] for c in combo], T, t)
            key = (ny(st), tuple(sorted(combo)))
            r, z = res.get(key, (0.0, 0.0))
            res[key] = (r + float(rat) * v, z + float(z3) * v)
    return res

if __name__ == '__main__':
    print(len(CT3), 'CT3 terms;', len(CJ3), 'CJ3 terms')
    for x in CT3: print(x)
    print(sum(1 for x in CJ3 if x[1] != 0), 'CJ3 terms with zeta3')
    for x in CJ3[:6] + CJ3[-8:]: print(x)
