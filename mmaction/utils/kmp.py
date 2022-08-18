def KMP(s, p):
    """
    s为主串
    p为模式串
    如果t里有p, 返回打头下标
    """
    nex = getNext(p)
    i = 0
    j = 0   # 分别是s和p的指针
    while i < len(s) and j < len(p):
        if j == -1 or s[i] == p[j]: # j==-1是由于j=next[j]产生
            i += 1
            j += 1
        else:
            j = nex[j]

    if j == len(p): # j走到了末尾，说明匹配到了
        return i - j
    else:
        return -1

def getNext(p):
    """
    p为模式串
    返回next数组, 即部分匹配表
    """
    nex = [0] * (len(p) + 1)
    nex[0] = -1
    i = 0
    j = -1
    while i < len(p):
        if j == -1 or p[i] == p[j]:
            i += 1
            j += 1
            nex[i] = j     # 这是最大的不同：记录next[i]
        else:
            j = nex[j]

    return nex


def bruteforce(input, target):
    if len(input) == 0:
        return []
    idx = 0
    input_label = []
    temp_idx = []
    input_idx = 0
    while idx < len(target):
        if input[input_idx] == target[idx]:
            temp_idx.append(idx)
            input_idx += 1
            idx += 1
        else:
            if input_idx == 0:
                idx += 1
            input_idx = 0
            temp_idx = []

        if input_idx == len(input):
            input_label.append(temp_idx)
            temp_idx = []
            input_idx = 0

    return input_label

# t = [1,1,2,3,1,2,4,6]
# p = [1,2,3]

# bruteforce(p, t)
# a = 1

