def get_openai_key():
    f = open('openai_key', 'r')
    key = f.read().strip()
    return key