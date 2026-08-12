import hashlib


def short_id(text: str, length: int=5):
    """
    Deterministic short ID from a string.
    Uses SHA-256 + base62 encoding
    :param text:
    :param length:
    :return:
    """
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    base = len(CHARS)

    hash_int = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
    n = hash_int % (base ** length)

    result = []
    for _ in range(length):
        result.append(CHARS[n % base])
        n //= base
    return "".join(reversed(result))

