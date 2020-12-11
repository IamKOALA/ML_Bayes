import math
import os
from matplotlib import pyplot as plt


def generate_ngram(header, message, n):
    ngrams = []

    header_words = header.split()
    msg_words = message.split()

    # append header n-grams
    if len(header_words) >= n:
        for i in range(len(header_words) - n + 1):
            gram = ""
            for j in range(n):
                gram += header_words[i + j] + " "
            gram = gram[: -1]
            ngrams.append(gram)

    # append msg n-grams
    if len(msg_words) >= n:
        for i in range(len(msg_words) - n + 1):
            gram = ""
            for j in range(n):
                gram += msg_words[i + j] + " "
            gram = gram[: -1]
            ngrams.append(gram)

    return ngrams


def naive_bayes(message_array, a):
    roc = []
    solution = 0
    optimal_n = 0
    for n in range(3):
        predicted = 0
        length = 0
        for part in range(len(message_array)):
            spam = {}
            legit = {}
            spam_cnt = 0

            for i in range(len(message_array)):
                if part != i:
                    length += len(message_array[i])
                    for msg in message_array[i]:
                        if msg[2]:
                            spam_cnt += 1
                        for gram in generate_ngram(msg[0], msg[1], n + 1):
                            if msg[2]:
                                if spam.keys().__contains__(gram):
                                    spam[gram] += 1
                                else:
                                    spam[gram] = 1
                            else:
                                if legit.keys().__contains__(gram):
                                    legit[gram] += 1
                                else:
                                    legit[gram] = 1

            d = spam_cnt / length

            sum_spam_cnt = sum(spam.values())
            sum_legit_cnt = sum(legit.values())

            unique = spam.copy()
            unique.update(legit)
            unique_cnt = len(unique)

            for msg in message_array[part]:
                ngrams = generate_ngram(msg[0], msg[1], n + 1)
                p_spam = math.log(d, math.e)
                p_legit = math.log(1 - d, math.e)

                for k in ngrams:
                    if spam.keys().__contains__(k):
                        p_spam += math.log((a + spam[k]) / (sum_spam_cnt + a * unique_cnt), math.e)
                    else:
                        p_spam += math.log(a / (sum_spam_cnt + a * unique_cnt), math.e)

                    if legit.keys().__contains__(k):
                        p_legit += math.log((a + legit[k]) / (sum_legit_cnt + a * unique_cnt), math.e)
                    else:
                        p_legit += math.log(a / (sum_legit_cnt + a * unique_cnt), math.e)

                    if msg[2]:
                        pair = (p_spam - p_legit, -1)
                    else:
                        pair = (p_spam - p_legit, 1)
                    roc.append(pair)
                if (msg[2] and p_spam > p_legit) or (not msg[2] and p_spam <= p_legit):
                    predicted += 1

        print(str(n) + " : " + str(predicted))
        print(length / 9)

        predicted = predicted * 9 / length
        print(predicted)

        if solution < predicted:
            solution = predicted
            optimal_n = n + 1

        roc = list(reversed(sorted(roc, key=lambda item: item[0])))

        fpr = [0.0] * (len(roc) + 1)
        tpr = [0.0] * (len(roc) + 1)

        lg_cnt = 0
        sp_cnt = 0

        for r in roc:
            if r[1] == 1:
                lg_cnt += 1
            else:
                sp_cnt += 1

        for k in range(len(roc)):
            if roc[k][1] == -1:
                fpr[k + 1] = fpr[k] + 1 / sp_cnt
                tpr[k + 1] = tpr[k]
            else:
                fpr[k + 1] = fpr[k]
                tpr[k + 1] = tpr[k] + 1 / lg_cnt

        plt.plot(tpr, fpr)
        plt.show()

    return optimal_n


def penalty(message_array, n, a):
    lam_spam = 1
    lam_legit = 1
    spam_remained = True
    x = []
    y = []

    while spam_remained:
        spam_cnt_remained = 0
        lam_legit *= 10 ** 8
        predicted = 0
        length = 0

        for part in range(len(message_array)):
            spam = {}
            legit = {}
            spam_cnt = 0

            for i in range(len(message_array)):
                if part != i:
                    length += len(message_array[i])
                    for msg in message_array[i]:
                        if msg[2]:
                            spam_cnt += 1
                        for gram in generate_ngram(msg[0], msg[1], n + 1):
                            if msg[2]:
                                if spam.keys().__contains__(gram):
                                    spam[gram] += 1
                                else:
                                    spam[gram] = 1
                            else:
                                if legit.keys().__contains__(gram):
                                    legit[gram] += 1
                                else:
                                    legit[gram] = 1

            d = spam_cnt / length

            sum_spam_cnt = sum(spam.values())
            sum_legit_cnt = sum(legit.values())

            unique = spam.copy()
            unique.update(legit)
            unique_cnt = len(unique)

            for msg in message_array[part]:
                ngrams = generate_ngram(msg[0], msg[1], n + 1)
                p_spam = math.log(d, math.e)
                p_legit = math.log(1 - d, math.e)

                for k in ngrams:
                    if spam.keys().__contains__(k):
                        p_spam += math.log((a + spam[k]) / (sum_spam_cnt + a * unique_cnt), math.e)
                    else:
                        p_spam += math.log(a / (sum_spam_cnt + a * unique_cnt), math.e)

                    if legit.keys().__contains__(k):
                        p_legit += math.log((a + legit[k]) / (sum_legit_cnt + a * unique_cnt), math.e)
                    else:
                        p_legit += math.log(a / (sum_legit_cnt + a * unique_cnt), math.e)
                p_spam += math.log(lam_spam, math.e)
                p_legit += math.log(lam_legit, math.e)
                if (msg[2] and p_spam > p_legit) or (not msg[2] and p_spam <= p_legit):
                    predicted += 1
                if not msg[2] and p_spam > p_legit:
                    spam_cnt_remained += 1
        if spam_cnt_remained == 0:
            spam_remained = False

        predicted = predicted * 9 / length

        print(str(predicted) + "  " + str(math.log(lam_legit, 10)))

        x.append(predicted)
        y.append(math.log(lam_legit, math.e))
    plt.plot(y, x)
    plt.show()


# def penalty(message_array, part, n, a):
#     lam_spam = 1
#     lam_legit = 1
#     spam_remained = True
#     x = []
#     y = []
#
#     while spam_remained:
#         spam_cnt_remained = 0
#         lam_legit *= 10
#         predicted = 0
#
#         spam = {}
#         legit = {}
#         spam_cnt = 0
#         length = 0
#         for i in range(len(message_array)):
#             if part != i:
#                 length += len(message_array[i])
#                 for msg in message_array[i]:
#                     if msg[2]:
#                         spam_cnt += 1
#                     for gram in generate_ngram(msg[0], msg[1], n + 1):
#                         if msg[2]:
#                             if spam.keys().__contains__(gram):
#                                 spam[gram] += 1
#                             else:
#                                 spam[gram] = 1
#                         else:
#                             if legit.keys().__contains__(gram):
#                                 legit[gram] += 1
#                             else:
#                                 legit[gram] = 1
#
#         d = spam_cnt / length
#
#         sum_spam_cnt = sum(spam.values())
#         sum_legit_cnt = sum(legit.values())
#
#         unique = spam.copy()
#         unique.update(legit)
#         unique_cnt = len(unique)
#
#         for msg in message_array[part]:
#             ngrams = generate_ngram(msg[0], msg[1], n + 1)
#             p_spam = math.log(d, math.e)
#             p_legit = math.log(1 - d, math.e)
#
#             for k in ngrams:
#                 if spam.keys().__contains__(k):
#                     p_spam += math.log((a + spam[k]) / (sum_spam_cnt + a * unique_cnt), math.e)
#                 else:
#                     p_spam += math.log(a / (sum_spam_cnt + a * unique_cnt), math.e)
#
#                 if legit.keys().__contains__(k):
#                     p_legit += math.log((a + legit[k]) / (sum_legit_cnt + a * unique_cnt), math.e)
#                 else:
#                     p_legit += math.log(a / (sum_legit_cnt + a * unique_cnt), math.e)
#             p_spam += math.log(lam_spam, math.e)
#             p_legit += math.log(lam_legit, math.e)
#             if (msg[2] and p_spam > p_legit) or (not msg[2] and p_spam <= p_legit):
#                 predicted += 1
#             if not msg[2] and p_spam > p_legit:
#                 spam_cnt_remained += 1
#         if spam_cnt_remained == 0:
#             spam_remained = False
#
#         print(predicted)
#         print(length)
#         print(lam_legit)
#
#         predicted = predicted / (len(message_array[part]))
#
#         x.append(predicted)
#         y.append(math.log(lam_legit, math.e))
#     plt.plot(y, x)
#     plt.show()


if __name__ == "__main__":
    message_array = []
    for part_num in range(10):
        part_array = []

        for file_message in os.listdir("part" + str(part_num + 1)):
            file = open("part" + str(part_num + 1) + "\\" + file_message)

            subject = file.readline()[9:-1]
            text = file.read()[1:-1]

            part_array.append([subject, text, True if "spmsg" in file_message else False])

        message_array.append(part_array)

    best_n = naive_bayes(message_array, 0.001)
    print(best_n)
    penalty(message_array, best_n, 0.001)
