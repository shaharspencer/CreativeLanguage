import matplotlib.pyplot as plt



if __name__ == '__main__':
    words = [token.text for token in doc if
             token.is_stop != True and token.is_punct != True]
