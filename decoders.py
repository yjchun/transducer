import math

def log_sum_exp(a, b):
    """
    Stable log sum exp.
    """
    return max(a, b) + math.log1p(math.exp(-abs(a-b)))


#TODO: it works but not passing test
def decode_static(log_probs, beam_size=1, blank=0):
    """
    Decode best prefix in the RNN Transducer. This decoder is static, it does
    not update the next step distribution based on the previous prediction. As
    such it looks for hypotheses which are length U.
    :param log_probs: FloatTensor(T,U,V) where T=encoder output, U=Predicted output, V=nbclasses
    """
    T, U, V = log_probs.shape
    beam = [[(), 1.0]] # list of best k so far.

    t = 0
    for u in range(U):
        while t < T:
            new_beam = {}
            # Pr(k|u,t)
            for hyp, score in beam:
                # find all probability
                for v in range(V):
                    if v == blank:
                        new_hyp = hyp
                        new_score = score + log_probs[t, u, v]
                    else:
                        new_hyp = hyp + (v,)
                        new_score = score + log_probs[t, u, v]

                    old_score = new_beam.get(new_hyp, None)
                    if old_score is not None:
                        new_beam[new_hyp] = log_sum_exp(old_score, new_score)
                    else:
                        new_beam[new_hyp] = new_score

            new_beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)
            beam = new_beam[:beam_size]
            maxlen = max([len(l[0]) for l in beam])
            if maxlen == u: # BLANK. advance T
                t += 1
                continue
            break

    hyp, score = beam[0]
    return hyp, score + log_probs[-1, -1, blank]


if __name__ == "__main__":
    import transducer.ref_transduce as rt
    import numpy as np
    np.random.seed(10)
    T = 10
    U = 5
    V = 5
    blank = V - 1
    beam_size = 500
    log_probs = np.random.randn(T, U, V)
    log_probs = rt.log_softmax(log_probs, axis=2)
    labels, beam_ll = decode_static(log_probs, beam_size, blank)
    print(labels)
    _, ll = rt.forward_pass(log_probs, labels, blank)
    assert np.allclose(ll, beam_ll, rtol=1e-9, atol=1e-9), \
            "Bad result from beam search."
