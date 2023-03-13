def pretty_fmt(num, suffix="B", base=1000.):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < base:
            return f"{num:.0f}{unit}{suffix}"
        num /= base
    return f"{num:.1f}Y{suffix}"
