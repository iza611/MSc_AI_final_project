def xor(a, b):
    return (a and not b) or (not a and b)

def gesture_synthesis(gestures_preds, threshold = 0.5):
    G = [(detection[5], detection[4]) for detection in gestures_preds]
 
    if len(G) == 0:
        p = 0
    elif len(G) == 1:
        g, c = G[0]
        p = g if c > threshold else 0
    else:
        G = sorted(G, key=lambda x: x[1], reverse=True)
        g_0 = G[0][0] if G[0][1] > threshold else 0
        g_1 = G[1][0] if G[1][1] > threshold else 0
            
        if g_0 == 0 and g_1 == 0:
            p = 0 
        elif xor(g_0!=0, g_1!=0):
            p = g_0 + g_1
        elif g_0 != g_1:
            p = 0
        elif g_0 == g_1:
            p = g_0
        else:
            raise Exception(f"Unexpected case of g_0={g_0} and g_1={g_1}.")
                    
    return p