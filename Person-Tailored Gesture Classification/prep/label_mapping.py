def class_to_label(cls):
    
    cls = int(cls)
    mapping = {
        0: 'no_gesture',
        1: 'like',
        2: 'ok',
        3: 'palm',
        4: 'two_up'
    }
    
    return mapping[cls]